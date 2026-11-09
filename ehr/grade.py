"""Grading: compare the agent's active actions with what the task rule requires.

Expected actions are computed from the environment's final state, so events
that fired during the run (a new lab result, new inbox messages) count, and
events that never fired do not.
"""

from __future__ import annotations

from core.checker import score_set
from ehr.env import EHREnv, ingredient
from ehr.records import days_between

POTASSIUM = "Potassium"
A1C = "Hemoglobin A1c/Hemoglobin.total in Blood"
K_THRESHOLD = 5.0
A1C_THRESHOLD = 6.5
VISIT_WINDOW_DAYS = 182


def latest_value(p: dict, name: str) -> float | None:
    rows = [o for o in p["observations"] if o["name"] == name]
    if not rows:
        return None
    last = max(rows, key=lambda o: o["date"])
    return float(last["value"])


def active_med(p: dict, ing: str) -> dict | None:
    meds = [m for m in p["medications"] if not m["stop"] and ingredient(m["name"]) == ing]
    return max(meds, key=lambda m: m["start"]) if meds else None


def has_penicillin_allergy(p: dict) -> bool:
    if any("penicillin" in a["substance"].lower() for a in p["allergies"]):
        return True
    return any("penicillin" in e["note"].lower() for e in p["encounters"])


def expected_update(env: EHREnv) -> set[tuple]:
    out = set()
    for pid, p in env.patients.items():
        v = latest_value(p, POTASSIUM)
        med = active_med(p, "lisinopril")
        if v is not None and v > K_THRESHOLD and med:
            out.add(("hold", pid, "lisinopril"))
            out.add(("message", med["prescriber"], pid))
    return out


def expected_distractor(env: EHREnv) -> set[tuple]:
    out = set()
    for pid, p in env.patients.items():
        v = latest_value(p, A1C)
        last = max(e["date"] for e in p["encounters"])
        if v is not None and v > A1C_THRESHOLD and days_between(last, env.today) > VISIT_WINDOW_DAYS:
            out.add(("schedule", pid, "Diabetes follow-up"))
    return out


def expected_distance(env: EHREnv) -> set[tuple]:
    out = set()
    for pid, p in env.patients.items():
        drug = "azithromycin" if has_penicillin_allergy(p) else "amoxicillin"
        out.add(("order", pid, drug))
    return out


def expected_interleave(env: EHREnv) -> set[tuple]:
    """Apply delivered inbox messages in order: requests add, amendments change or cancel."""
    by_msg: dict[str, tuple] = {}
    for m in env.inbox:
        if not m["delivered"]:
            continue
        act = m["_action"]
        if act["op"] == "request":
            by_msg[m["id"]] = tuple(act["key"])
        elif act["op"] == "replace":
            by_msg[act["target"]] = tuple(act["key"])
        elif act["op"] == "cancel":
            by_msg.pop(act["target"], None)
    return set(by_msg.values())


RULES = {
    "update": expected_update,
    "distractor": expected_distractor,
    "distance": expected_distance,
    "interleave": expected_interleave,
}


def expected_actions(env: EHREnv) -> set[tuple]:
    return RULES[env.task["family"]](env)


def grade(env: EHREnv) -> dict:
    gold = expected_actions(env)
    pred = env.active_actions()
    s = score_set({"|".join(k) for k in pred}, {"|".join(k) for k in gold})
    return {
        **s,
        "missing": sorted("|".join(k) for k in gold - pred),
        "extra": sorted("|".join(k) for k in pred - gold),
        "events_fired": sum(1 for ev in env.events if ev["fired"]),
        "events_total": len(env.events),
        "finished": env.done,
    }
