"""Load Synthea CSV tables into structured, JSON-serializable patient records.

Only synthetic data is used. Names, addresses, and ID numbers are dropped.
Everything after TODAY is dropped so the system has a consistent present.

Patient record layout (all dates are YYYY-MM-DD):
    {
      "id": "P0001", "sex": "F", "birth_date": "1970-01-01",
      "encounters":   [{"id", "date", "class", "description", "provider", "reason", "note"}],
      "conditions":   [{"date", "description", "resolved"}],
      "medications":  [{"id", "name", "start", "stop", "prescriber", "reason"}],
      "observations": [{"date", "name", "value", "units"}],
      "allergies":    [{"date", "substance", "reaction"}],
    }
Provider fields hold Synthea provider UUIDs until a task maps them to short IDs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

TODAY = "2021-12-01"


def days_before(n: int, ref: str = TODAY) -> str:
    return (date.fromisoformat(ref) - timedelta(days=n)).isoformat()


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def age_on(birth: str, ref: str = TODAY) -> int:
    b, r = date.fromisoformat(birth), date.fromisoformat(ref)
    return r.year - b.year - ((r.month, r.day) < (b.month, b.day))


def _rows(path: Path):
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def _d(ts: str) -> str:
    return ts[:10] if ts else ""


def load_providers(csv_dir: str | Path) -> dict[str, str]:
    """Provider UUID -> specialty (title case)."""
    out = {}
    for r in _rows(Path(csv_dir) / "providers.csv"):
        out[r["Id"]] = r.get("SPECIALITY", "").title() or "General Practice"
    return out


def load_records(csv_dir: str | Path, today: str = TODAY) -> list[dict]:
    """Load living patients aged 18 to 95 with at least 5 encounters before `today`."""
    d = Path(csv_dir)
    raw = sorted(_rows(d / "patients.csv"), key=lambda r: r["Id"])
    pats: dict[str, dict] = {}
    for i, r in enumerate(raw, start=1):
        pats[r["Id"]] = {
            "id": f"P{i:04d}",
            "sex": r["GENDER"],
            "birth_date": r["BIRTHDATE"],
            "death_date": r["DEATHDATE"] or None,
            "encounters": [],
            "conditions": [],
            "medications": [],
            "observations": [],
            "allergies": [],
        }

    enc_provider: dict[str, str] = {}
    for r in _rows(d / "encounters.csv"):
        p = pats.get(r["PATIENT"])
        dt = _d(r["START"])
        if not p or dt > today:
            continue
        enc_provider[r["Id"]] = r["PROVIDER"]
        p["encounters"].append(
            {
                "id": r["Id"],
                "date": dt,
                "class": r["ENCOUNTERCLASS"],
                "description": r["DESCRIPTION"],
                "provider": r["PROVIDER"],
                "reason": r.get("REASONDESCRIPTION", ""),
                "note": "",
            }
        )

    for r in _rows(d / "conditions.csv"):
        p = pats.get(r["PATIENT"])
        if not p or r["START"] > today:
            continue
        stop = r["STOP"] if r["STOP"] and r["STOP"] <= today else ""
        p["conditions"].append({"date": r["START"], "description": r["DESCRIPTION"], "resolved": stop})

    for r in _rows(d / "medications.csv"):
        p = pats.get(r["PATIENT"])
        start = _d(r["START"])
        if not p or start > today:
            continue
        stop = _d(r["STOP"])
        p["medications"].append(
            {
                "id": "",
                "name": r["DESCRIPTION"],
                "start": start,
                "stop": stop if stop and stop <= today else "",
                "prescriber": enc_provider.get(r["ENCOUNTER"], ""),
                "reason": r.get("REASONDESCRIPTION", ""),
            }
        )

    for r in _rows(d / "observations.csv"):
        p = pats.get(r["PATIENT"])
        dt = _d(r["DATE"])
        if not p or dt > today or r.get("TYPE") == "text":
            continue
        p["observations"].append({"date": dt, "name": r["DESCRIPTION"], "value": r["VALUE"], "units": r["UNITS"]})

    for r in _rows(d / "allergies.csv"):
        p = pats.get(r["PATIENT"])
        if not p or r["START"] > today:
            continue
        p["allergies"].append(
            {"date": r["START"], "substance": r["DESCRIPTION"], "reaction": r.get("DESCRIPTION1", "")}
        )

    out = []
    for p in pats.values():
        age = age_on(p["birth_date"], today)
        if p["death_date"] or not 18 <= age <= 95 or len(p["encounters"]) < 5:
            continue
        del p["death_date"]
        for key, field in [("encounters", "date"), ("conditions", "date"), ("medications", "start"),
                           ("observations", "date"), ("allergies", "date")]:
            p[key].sort(key=lambda x: x[field])
        out.append(p)
    return sorted(out, key=lambda p: p["id"])


def record_chars(p: dict) -> int:
    """Rough size of a patient record, used to keep environments the same size."""
    return sum(len(str(x)) for k in ("encounters", "conditions", "medications", "observations", "allergies")
               for x in p[k])
