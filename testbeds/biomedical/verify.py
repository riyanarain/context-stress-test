"""Check biomedical gold answers using only the rendered chart text.

This re-derives every condition by reading the chart files the agent will
see, without the structured data used to make the gold answers. If the two
ever disagree, the chart is missing or misstating something.

Usage:
    python -m testbeds.biomedical.verify --data data/biomedical
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from core.docstore import DocStore
from core.task import load_tasks
from testbeds.biomedical.generate import LAB_THRESHOLDS

OBS_NAMES = {
    "39156-5": "Body Mass Index",
    "8480-6": "Systolic Blood Pressure",
    "4548-4": "Hemoglobin A1c/Hemoglobin.total in Blood",
    "2093-3": "Total Cholesterol",
    "18262-6": "Low Density Lipoprotein Cholesterol",
}
DATE = re.compile(r"^(\d{4}-\d{2}-\d{2}) ")


def holds_from_text(text: str, cond: dict) -> bool:
    kind, key, pos = cond["kind"], cond["key"], cond["positive"]
    lines = text.splitlines()
    if kind == "diagnosis":
        val = any(
            ln.strip() == f"Diagnosis: {key}" or ln.strip().startswith(f"Diagnosis: {key} (resolved")
            for ln in lines
        )
    elif kind == "medication":
        pat = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
        val = any(ln.strip().startswith("Medication:") and pat.search(ln.split(", started")[0]) for ln in lines)
    elif kind == "lab":
        name = OBS_NAMES[key]
        cur_date, latest = None, None
        for ln in lines:
            m = DATE.match(ln)
            if m:
                cur_date = m.group(1)
            s = ln.strip()
            s = DATE.sub("", s) if DATE.match(s) else s
            if s.startswith(f"Observation: {name} = "):
                v = float(s.split(" = ", 1)[1].split()[0])
                if latest is None or cur_date >= latest[0]:
                    latest = (cur_date, v)
        val = latest is not None and latest[1] > LAB_THRESHOLDS[key]
    elif kind == "sex":
        val = f"Sex: {key}" in lines
    elif kind == "birth":
        year = int(next(ln for ln in lines if ln.startswith("Birth date:")).split(":")[1].strip()[:4])
        val = year < int(key)
    elif kind == "deceased":
        val = not next(ln for ln in lines if ln.startswith("Death date:")).endswith("(living)")
    else:
        raise ValueError(kind)
    return val if pos else not val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/biomedical")
    args = ap.parse_args()
    store = DocStore(args.data)
    tasks = load_tasks(Path(args.data) / "tasks.jsonl")
    cache: dict[str, str] = {}
    n_checked = n_bad = 0
    for t in tasks:
        prev = set(t.doc_ids)
        for s in t.steps:
            for d in prev:
                text = cache.setdefault(d, store.read(d))
                ok = holds_from_text(text, s.condition) == (d in s.gold)
                n_checked += 1
                if not ok:
                    n_bad += 1
                    if n_bad <= 10:
                        print(f"mismatch: {t.task_id} {d} {s.condition}")
            prev = set(s.gold)
    print(f"checked {n_checked} (patient, step) pairs, {n_bad} mismatches")
    raise SystemExit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
