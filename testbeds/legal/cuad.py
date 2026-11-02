"""Load CUAD contracts and their clause annotations.

CUAD (Hendrycks et al., NeurIPS 2021) has 510 commercial contracts annotated
by lawyers for 41 clause categories, released in SQuAD format. We keep, per
contract, the full text, which categories are present, and a normalized
governing-law jurisdiction parsed from the Governing Law annotation.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Contract:
    doc_id: str
    title: str
    text: str
    clauses: dict[str, list[str]] = field(default_factory=dict)  # category -> annotated spans
    governing_law: str | None = None  # normalized jurisdiction, None if no clause
    governing_law_unparsed: bool = False  # clause present but jurisdiction not recognized

    def has(self, category: str) -> bool:
        return bool(self.clauses.get(category))


US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming",
]

# Non-US jurisdictions: canonical name -> regex alternatives.
OTHER_JURISDICTIONS = {
    "District of Columbia": [r"District of Columbia"],
    "England and Wales": [r"England(?: and Wales)?", r"English", r"United Kingdom"],
    "Scotland": [r"Scotland", r"Scottish"],
    "Ontario": [r"Ontario"],
    "British Columbia": [r"British Columbia"],
    "Alberta": [r"Alberta"],
    "Quebec": [r"Quebec"],
    "China": [r"People'?s Republic of China", r"\bPRC\b", r"\bChina\b"],
    "Hong Kong": [r"Hong Kong"],
    "Japan": [r"Japan"],
    "Singapore": [r"Singapore"],
    "Israel": [r"Israel"],
    "Germany": [r"Germany", r"German"],
    "Switzerland": [r"Switzerland", r"Swiss"],
    "Netherlands": [r"Netherlands", r"Dutch"],
    "Ireland": [r"Ireland", r"Irish"],
    "India": [r"\bIndia\b"],
    "Korea": [r"Korea"],
    "Sweden": [r"Sweden", r"Swedish"],
    "France": [r"France", r"French"],
    "Taiwan": [r"Taiwan"],
    "Australia": [r"Australia", r"New South Wales", r"Victoria, Australia"],
    "Cayman Islands": [r"Cayman Islands"],
    "Bermuda": [r"Bermuda"],
    "Luxembourg": [r"Luxembourg"],
    "Denmark": [r"Denmark", r"Danish"],
    "Austria": [r"Austria"],
    "Italy": [r"Italy", r"Italian"],
    "Spain": [r"Spain"],
    "Malaysia": [r"Malaysia"],
    "Thailand": [r"Thailand"],
    "Philippines": [r"Philippines"],
    "Nova Scotia": [r"Nova Scotia"],
    "Kazakhstan": [r"Kazakhstan"],
    "Papua New Guinea": [r"Papua New Guinea"],
    "Colombia": [r"Colombia"],
    "South Africa": [r"South Africa"],
    "United States (federal)": [r"United States of America"],
}


def _jurisdiction_patterns() -> list[tuple[str, re.Pattern]]:
    pats = []
    for st in US_STATES:
        pats.append((st, re.compile(rf"\b{st}\b", re.IGNORECASE)))
    for name, alts in OTHER_JURISDICTIONS.items():
        for a in alts:
            pat = a if a.startswith(r"\b") else rf"\b{a}\b"
            pats.append((name, re.compile(pat, re.IGNORECASE)))
    return pats


_PATTERNS = _jurisdiction_patterns()
_LAW_WORD = re.compile(r"\blaws?\b", re.IGNORECASE)


def parse_governing_law(span: str) -> str | None:
    """Return the jurisdiction whose law governs, from a Governing Law span.

    Clauses often name a second jurisdiction for courts or venue, so we take
    the first jurisdiction mentioned after the word 'law'. Overlapping matches
    (Virginia inside West Virginia) resolve to the longest match.
    """
    span = re.sub(r"\s+", " ", span)
    matches = [
        (m.start(), m.end(), name) for name, pat in _PATTERNS for m in pat.finditer(span)
    ]
    # Drop matches contained in a longer match ("Virginia" inside "West Virginia").
    matches = [
        a for a in matches
        if not any(b[0] <= a[0] and a[1] <= b[1] and (b[1] - b[0]) > (a[1] - a[0]) for b in matches)
    ]
    if not matches:
        return None
    law = _LAW_WORD.search(span)
    start = law.start() if law else 0
    after = [m for m in matches if m[0] >= start]
    chosen = min(after or matches)
    return chosen[2]


def load_category_descriptions(path: str | Path) -> dict[str, str]:
    out = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].startswith("Category:"):
                continue
            cat = row[0].removeprefix("Category:").strip()
            desc = row[1].removeprefix("Description:").strip()
            out[cat.lower()] = desc
    return out


def load_cuad(json_path: str | Path) -> list[Contract]:
    """Load CUADv1.json. Contracts get stable IDs C0001..C0510 sorted by title."""
    data = json.loads(Path(json_path).read_text())["data"]
    data = sorted(data, key=lambda c: c["title"])
    contracts = []
    for i, c in enumerate(data, start=1):
        p = c["paragraphs"][0]
        clauses = {}
        for qa in p["qas"]:
            cat = qa["id"].split("__", 1)[1]
            clauses[cat] = [a["text"] for a in qa["answers"]]
        gl_spans = clauses.get("Governing Law", [])
        gl, unparsed = None, False
        if gl_spans:
            gl = parse_governing_law(" ".join(gl_spans))
            unparsed = gl is None
        contracts.append(
            Contract(
                doc_id=f"C{i:04d}",
                title=c["title"],
                text=p["context"],
                clauses=clauses,
                governing_law=gl,
                governing_law_unparsed=unparsed,
            )
        )
    return contracts
