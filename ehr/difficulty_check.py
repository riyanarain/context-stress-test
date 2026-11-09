"""Check that each difficulty knob does what it claims, using simple scripted strategies.

Each strategy has one specific weakness. It should do well at level 0 and
worse at the levels that target that weakness. No model is involved.

    update      earliest-value: uses the oldest potassium instead of the latest
                no-recheck: reads every patient once, acts, never looks again
    distractor  off-by-one: uses A1c >= 6.5 and >= 182 days instead of > and >
    distance    truncated-read: checks the allergy list plus the first 3000
                characters of get_encounters (as a truncation policy would)
    interleave  no-amendments: acts on the original requests only
                no-recheck: reads the inbox once at the start

Usage:
    python -m ehr.difficulty_check --tasks data/ehr/tasks
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from ehr.env import FORMULARY, EHREnv, ingredient
from ehr.grade import A1C, POTASSIUM, grade
from ehr.records import days_between


def _act(env: EHREnv, keys: set[tuple]) -> None:
    for key in sorted(keys):
        kind = key[0]
        if kind == "hold":
            env.call("hold_medication", {"patient_id": key[1], "medication": key[2]})
        elif kind == "message":
            env.call("send_message", {"provider_id": key[1], "patient_id": key[2], "text": "fyi"})
        elif kind == "schedule":
            env.call("schedule_appointment", {"patient_id": key[1], "appointment_type": key[2]})
        elif kind == "order":
            drug = next(f for f in FORMULARY if ingredient(f) == key[2])
            env.call("order_medication", {"patient_id": key[1], "medication": drug})


def _k_values(env: EHREnv, pid: str) -> list[tuple[str, float]]:
    out = env.call("get_labs", {"patient_id": pid, "test_name": "potassium"})
    return [(d, float(v)) for d, v in re.findall(r"(\d{4}-\d{2}-\d{2})  Potassium  ([\d.]+)", out)]


def _lisinopril_prescriber(env: EHREnv, pid: str) -> str | None:
    out = env.call("get_medications", {"patient_id": pid})
    m = re.search(r"lisinopril[^\n]*active[^\n]*prescriber (DR\d+)", out)
    return m.group(1) if m else None


def update_strategy(env: EHREnv, which: str) -> None:
    keys = set()
    for pid in sorted(env.patients):
        vals = _k_values(env, pid)
        if not vals:
            continue
        v = min(vals)[1] if which == "earliest" else max(vals)[1]
        if v > 5.0:
            keys |= {("hold", pid, "lisinopril"), ("message", _lisinopril_prescriber(env, pid), pid)}
    _act(env, keys)  # no-recheck: never reads again, so alerts after acting are ignored


def distractor_off_by_one(env: EHREnv) -> None:
    keys = set()
    for pid, p in env.patients.items():
        a1c = max((o for o in p["observations"] if o["name"] == A1C), key=lambda o: o["date"])
        last = max(e["date"] for e in p["encounters"])
        if float(a1c["value"]) >= 6.5 and days_between(last, env.today) >= 182:
            keys.add(("schedule", pid, "Diabetes follow-up"))
    _act(env, keys)


def distance_truncated(env: EHREnv, limit: int = 3000) -> None:
    keys = set()
    for pid in sorted(env.patients):
        allergies = env.call("get_allergies", {"patient_id": pid}).lower()
        enc = env.call("get_encounters", {"patient_id": pid})[:limit].lower()
        allergic = "penicillin" in allergies or "penicillin" in enc
        keys.add(("order", pid, "azithromycin" if allergic else "amoxicillin"))
    _act(env, keys)


def _parse_request(text: str) -> tuple | None:
    m = re.match(r"Please schedule an? (.+) for patient (P\d+)\.", text)
    if m:
        return ("schedule", m.group(2), m.group(1))
    m = re.match(r"Please hold (\S+) for patient (P\d+)\.", text)
    if m:
        return ("hold", m.group(2), m.group(1))
    m = re.match(r"Please order (.+) for patient (P\d+)\.", text)
    if m:
        return ("order", m.group(2), ingredient(m.group(1)))
    return None


def interleave_strategy(env: EHREnv, which: str) -> None:
    inbox = env.call("read_inbox", {})
    msgs = re.findall(r"(MSG\d+)  \S+  from \S+: (.+)", inbox)
    by_id = {}
    for mid, text in msgs:
        req = _parse_request(text)
        if req:
            by_id[mid] = req
        elif which == "apply-amendments":
            m = re.match(r"Update to my message (MSG\d+): for patient (P\d+), please book an? (.+) instead", text)
            if m:
                by_id[m.group(1)] = ("schedule", m.group(2), m.group(3))
            m = re.match(r"Please disregard my message (MSG\d+)", text)
            if m:
                by_id.pop(m.group(1), None)
    _act(env, set(by_id.values()))  # never rereads the inbox


STRATEGIES = {
    "update": {"earliest-value": lambda e: update_strategy(e, "earliest"),
               "no-recheck (latest value)": lambda e: update_strategy(e, "latest")},
    "distractor": {"off-by-one thresholds": distractor_off_by_one},
    "distance": {"truncated read (3000 chars)": distance_truncated},
    "interleave": {"ignores amendments": lambda e: interleave_strategy(e, "ignore"),
                   "reads inbox once": lambda e: interleave_strategy(e, "apply-amendments")},
}


def run(task_dir: str | Path) -> dict:
    scores = defaultdict(list)
    for f in sorted(Path(task_dir).glob("*.json")):
        t = json.loads(f.read_text())
        for name, fn in STRATEGIES[t["family"]].items():
            env = EHREnv(t)
            fn(env)
            scores[(t["family"], name, t["level"])].append(grade(env)["exact"])
    return {k: statistics.mean(v) for k, v in scores.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/ehr/tasks")
    args = ap.parse_args()
    res = run(args.tasks)
    rows = sorted({(f, s) for f, s, _ in res})
    print("family | strategy | L0 | L1 | L2  (fraction of tasks fully correct)")
    for fam, strat in rows:
        cells = [f"{res.get((fam, strat, lvl), float('nan')):.2f}" for lvl in (0, 1, 2)]
        print(f"{fam} | {strat} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
