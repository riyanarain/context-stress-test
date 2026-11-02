"""Load Synthea synthetic patients and render each one as a text chart.

Synthea (MITRE) generates realistic but fully synthetic patient histories, so
there is no real patient data and no access restrictions. We use the CSV
sample release from github.com/synthetichealth/synthea-sample-data, which has
about 1,160 patients.

Each patient becomes one document: a header (ID, sex, birth date, death date)
followed by a dated log of encounters with their diagnoses, medications,
observations, procedures, immunizations, allergies, and care plans. Names,
addresses, and ID numbers from the source are left out. Billing tables are
skipped.

The structured facts used for gold answers are computed from the same rows
that produce the text, so every gold answer can be found in the chart.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Labs and vitals used in "most recent value" conditions.
# code -> (short name used in questions, unit)
TRACKED_OBSERVATIONS = {
    "39156-5": ("BMI", "kg/m2"),
    "8480-6": ("systolic blood pressure", "mm[Hg]"),
    "4548-4": ("hemoglobin A1c", "%"),
    "2093-3": ("total cholesterol", "mg/dL"),
    "18262-6": ("LDL cholesterol", "mg/dL"),
}


@dataclass
class Patient:
    doc_id: str
    source_id: str
    sex: str  # "F" or "M"
    birth_date: str
    death_date: str | None
    text: str = ""
    diagnoses: set[str] = field(default_factory=set)  # exact description strings
    medications: set[str] = field(default_factory=set)  # exact description strings
    latest_obs: dict[str, float] = field(default_factory=dict)  # code -> value at latest date
    ambiguous_obs: set[str] = field(default_factory=set)  # codes with conflicting latest values

    @property
    def birth_year(self) -> int:
        return int(self.birth_date[:4])

    @property
    def deceased(self) -> bool:
        return bool(self.death_date)

    def takes(self, ingredient: str) -> bool:
        pat = re.compile(rf"\b{re.escape(ingredient)}\b", re.IGNORECASE)
        return any(pat.search(m) for m in self.medications)


def _rows(path: Path):
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def _day(ts: str) -> str:
    return ts[:10] if ts else ""


def _fmt_value(value: str, units: str) -> str:
    try:
        v = float(value)
        value = f"{v:g}"
    except ValueError:
        pass
    return f"{value} {units}".strip()


def load_synthea(csv_dir: str | Path) -> list[Patient]:
    """Load patients and render charts. IDs are P0001.. sorted by source ID."""
    d = Path(csv_dir)
    pats = sorted(_rows(d / "patients.csv"), key=lambda r: r["Id"])
    patients = {
        r["Id"]: Patient(
            doc_id=f"P{i:04d}",
            source_id=r["Id"],
            sex=r["GENDER"],
            birth_date=r["BIRTHDATE"],
            death_date=r["DEATHDATE"] or None,
        )
        for i, r in enumerate(pats, start=1)
    }

    encounters: dict[str, dict] = {}
    by_patient_enc: dict[str, list[str]] = defaultdict(list)
    for r in _rows(d / "encounters.csv"):
        if r["PATIENT"] not in patients:
            continue
        encounters[r["Id"]] = r
        by_patient_enc[r["PATIENT"]].append(r["Id"])

    # Lines attached to an encounter: enc_id -> list of (sort key, line)
    enc_lines: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # Lines with no matching encounter: patient -> list of (date, line)
    loose: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def attach(pid: str, enc: str, date: str, order: str, line: str) -> None:
        if enc in encounters:
            enc_lines[enc].append((order, line))
        else:
            loose[pid].append((date, line))

    for r in _rows(d / "conditions.csv"):
        p = patients.get(r["PATIENT"])
        if not p:
            continue
        p.diagnoses.add(r["DESCRIPTION"])
        line = f"  Diagnosis: {r['DESCRIPTION']}"
        if r["STOP"]:
            line += f" (resolved {_day(r['STOP'])})"
        attach(r["PATIENT"], r["ENCOUNTER"], _day(r["START"]), "1", line)

    for r in _rows(d / "medications.csv"):
        p = patients.get(r["PATIENT"])
        if not p:
            continue
        p.medications.add(r["DESCRIPTION"])
        line = f"  Medication: {r['DESCRIPTION']}"
        if r.get("REASONDESCRIPTION"):
            line += f" for {r['REASONDESCRIPTION']}"
        line += f", started {_day(r['START'])}"
        if r["STOP"]:
            line += f", stopped {_day(r['STOP'])}"
        attach(r["PATIENT"], r["ENCOUNTER"], _day(r["START"]), "2", line)

    obs_hist: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for r in _rows(d / "observations.csv"):
        if r["PATIENT"] not in patients:
            continue
        line = f"  Observation: {r['DESCRIPTION']} = {_fmt_value(r['VALUE'], r['UNITS'])}"
        attach(r["PATIENT"], r["ENCOUNTER"], _day(r["DATE"]), "3", line)
        if r["CODE"] in TRACKED_OBSERVATIONS:
            obs_hist[(r["PATIENT"], r["CODE"])].append((r["DATE"], r["VALUE"]))

    for r in _rows(d / "procedures.csv"):
        if r["PATIENT"] not in patients:
            continue
        line = f"  Procedure: {r['DESCRIPTION']}"
        if r.get("REASONDESCRIPTION"):
            line += f" for {r['REASONDESCRIPTION']}"
        date = _day(r.get("START") or r.get("DATE", ""))
        attach(r["PATIENT"], r["ENCOUNTER"], date, "4", line)

    for r in _rows(d / "immunizations.csv"):
        if r["PATIENT"] not in patients:
            continue
        attach(r["PATIENT"], r["ENCOUNTER"], _day(r["DATE"]), "5", f"  Immunization: {r['DESCRIPTION']}")

    for r in _rows(d / "allergies.csv"):
        if r["PATIENT"] not in patients:
            continue
        attach(r["PATIENT"], r["ENCOUNTER"], _day(r["START"]), "6", f"  Allergy: {r['DESCRIPTION']}")

    for r in _rows(d / "careplans.csv"):
        if r["PATIENT"] not in patients:
            continue
        line = f"  Care plan: {r['DESCRIPTION']}"
        if r.get("REASONDESCRIPTION"):
            line += f" for {r['REASONDESCRIPTION']}"
        attach(r["PATIENT"], r["ENCOUNTER"], _day(r["START"]), "7", line)

    # Most recent value per tracked observation. The chart shows days, not
    # times, so two different values on the latest day count as ambiguous.
    for (pid, code), hist in obs_hist.items():
        latest = max(_day(t) for t, _ in hist)
        vals = {v for t, v in hist if _day(t) == latest}
        p = patients[pid]
        try:
            fvals = {float(v) for v in vals}
        except ValueError:
            p.ambiguous_obs.add(code)
            continue
        if len(fvals) > 1:
            p.ambiguous_obs.add(code)
        else:
            p.latest_obs[code] = fvals.pop()

    for pid, p in patients.items():
        lines = [
            f"Patient {p.doc_id}",
            f"Sex: {p.sex}",
            f"Birth date: {p.birth_date}",
            f"Death date: {p.death_date or 'none (living)'}",
            "",
            "Encounters (oldest first):",
        ]
        encs = sorted(by_patient_enc[pid], key=lambda e: (encounters[e]["START"], e))
        for e in encs:
            r = encounters[e]
            head = f"{_day(r['START'])} {r['ENCOUNTERCLASS']} encounter: {r['DESCRIPTION']}"
            if r.get("REASONDESCRIPTION"):
                head += f" (reason: {r['REASONDESCRIPTION']})"
            lines.append("")
            lines.append(head)
            lines += [ln for _, ln in sorted(enc_lines[e], key=lambda x: x[0])]
        if loose[pid]:
            lines += ["", "Other records:"]
            lines += [f"{date} {ln.strip()}" for date, ln in sorted(loose[pid])]
        p.text = "\n".join(lines) + "\n"

    return sorted(patients.values(), key=lambda p: p.doc_id)
