"""Task families. Each family varies one difficulty factor at three levels
(0 = easy, 1 = medium, 2 = hard) while the environment size stays about the same.

update      Updated facts. Hold lisinopril and message the prescriber when the
            most recent potassium is above 5.0.
              0: one potassium value per patient
              1: several values; earlier values sit on the other side of 5.0
              2: as 1, plus new results posted mid-task that flip two patients
distractor  Similar distractors. Schedule a diabetes follow-up when the most
            recent A1c is above 6.5 and the last encounter was over 182 days ago.
              0: non-qualifying patients are far from both thresholds
              1: closer
              2: near-misses (A1c 6.4 or 6.5, last visit 176 to 182 days ago)
distance    Distance to a needed fact. Order amoxicillin for strep, or
            azithromycin if there is a penicillin allergy.
              0: allergy is in the allergy list
              1: allergy is only in a clinician note from 6 to 18 months ago
              2: allergy is only in a clinician note from 8 to 15 years ago
interleave  Interleaved goals. Carry out requests from the clinician inbox.
              0: five independent requests
              1: plus two later messages that change or cancel earlier requests
              2: the two amendments arrive mid-task, after the third write action
"""

from __future__ import annotations

import copy
import random

from ehr.env import APPOINTMENT_TYPES, FORMULARY, ingredient
from ehr.grade import A1C, POTASSIUM
from ehr.records import TODAY, days_before, record_chars

FAMILIES = ["update", "distractor", "distance", "interleave"]

INSTRUCTIONS = {
    "update": (
        "Every patient in the system is being discharged today and has an active lisinopril prescription. "
        "For each patient whose most recent potassium result is above 5.0 mmol/L, hold their lisinopril and send a "
        "message to the clinician who prescribed that lisinopril. Do not hold lisinopril for anyone else. "
        "New results can be posted while you work. Your final actions should match the most recent results, so "
        "cancel any action that no longer applies."
    ),
    "distractor": (
        "Schedule a Diabetes follow-up appointment for every patient whose most recent hemoglobin A1c is above 6.5% "
        "and whose most recent encounter was more than 182 days before today. Do not schedule anyone else."
    ),
    "distance": (
        "Every patient in the system was diagnosed with streptococcal sore throat today. Order one antibiotic for "
        "each patient: amoxicillin 500 MG oral capsule, unless the patient has a documented penicillin allergy, in "
        "which case order azithromycin 250 MG oral tablet. Allergies may be documented in the allergy list or in "
        "clinician notes on past encounters."
    ),
    "interleave": (
        "Work through the clinician inbox and carry out every request. Later messages may change or cancel earlier "
        "requests. Your final actions should match the latest instructions, so cancel earlier actions if needed. New "
        "messages can arrive while you work, so check the inbox again before you finish."
    ),
}

FILLER_NOTES = [
    "Discussed diet and exercise.",
    "Patient tolerated the visit well.",
    "Reviewed home medications; no changes.",
    "Reports seasonal allergies to pollen.",
    "Advised to return if symptoms worsen.",
    "Vaccination history reviewed.",
    "Blood pressure discussed; continue current plan.",
]
ALLERGY_NOTE = "Patient reports a history of hives after taking penicillin. Avoid penicillin-class antibiotics."


# ---------------------------------------------------------------- helpers
def _split(n: int, frac: float, rng: random.Random) -> list[bool]:
    """n booleans with about frac True, at least one of each when n >= 2."""
    k = min(max(1, round(n * frac)), n - 1) if n >= 2 else 1
    flags = [True] * k + [False] * (n - k)
    rng.shuffle(flags)
    return flags


def _provider_near(p: dict, day: str) -> str:
    before = [e for e in p["encounters"] if e["date"] <= day and e["provider"]]
    pool = before or [e for e in p["encounters"] if e["provider"]]
    return max(pool, key=lambda e: e["date"])["provider"]


def _truncate_after(p: dict, day: str) -> None:
    """Drop everything recorded after `day`, as if the chart ended then."""
    p["encounters"] = [e for e in p["encounters"] if e["date"] <= day]
    p["conditions"] = [c for c in p["conditions"] if c["date"] <= day]
    for c in p["conditions"]:
        if c["resolved"] and c["resolved"] > day:
            c["resolved"] = ""
    p["medications"] = [m for m in p["medications"] if m["start"] <= day]
    for m in p["medications"]:
        if m["stop"] and m["stop"] > day:
            m["stop"] = ""
    p["observations"] = [o for o in p["observations"] if o["date"] <= day]
    p["allergies"] = [a for a in p["allergies"] if a["date"] <= day]


def _new_encounter(p: dict, day: str, cls: str, desc: str, reason: str = "", note: str = "",
                   provider: str | None = None) -> dict:
    e = {"id": "", "date": day, "class": cls, "description": desc, "provider": provider or _provider_near(p, day),
         "reason": reason, "note": note}
    p["encounters"].append(e)
    return e


def select_patients(pool: list[dict], n: int, rng: random.Random, max_obs: int) -> list[dict]:
    pats = copy.deepcopy(rng.sample(pool, n))
    for p in pats:
        p["observations"] = sorted(p["observations"], key=lambda o: o["date"])[-max_obs:]
    return pats


def finalize(pats: list[dict], family: str, level: int, task_id: str, inbox=None, events=None,
             provider_specialty: dict[str, str] | None = None) -> dict:
    """Assign short encounter, medication, and provider IDs and build the task dict."""
    provider_specialty = provider_specialty or {}
    prov_map: dict[str, str] = {}

    def pid_of(uuid: str) -> str:
        if uuid not in prov_map:
            prov_map[uuid] = f"DR{len(prov_map) + 1:03d}"
        return prov_map[uuid]

    e_n = m_n = 0
    for p in sorted(pats, key=lambda p: p["id"]):
        for key, field in [("encounters", "date"), ("conditions", "date"), ("medications", "start"),
                           ("observations", "date"), ("allergies", "date")]:
            p[key].sort(key=lambda x: x[field])
        for e in p["encounters"]:
            e_n += 1
            e["id"] = f"E{e_n:05d}"
            e["provider"] = pid_of(e["provider"])
        for m in p["medications"]:
            m_n += 1
            m["id"] = f"M{m_n:05d}"
            if not m["prescriber"]:
                m["prescriber"] = _provider_near_mapped(p, m["start"])
            elif not m["prescriber"].startswith("DR"):
                m["prescriber"] = pid_of(m["prescriber"])
    for m in inbox or []:
        if not m["from"].startswith("DR"):
            m["from"] = pid_of(m["from"])
    providers = {short: provider_specialty.get(uuid, "General Practice") for uuid, short in prov_map.items()}
    env_chars = sum(record_chars(p) for p in pats)
    return {
        "task_id": task_id,
        "family": family,
        "level": level,
        "today": TODAY,
        "instructions": INSTRUCTIONS[family],
        "patients": {p["id"]: p for p in sorted(pats, key=lambda p: p["id"])},
        "providers": providers,
        "inbox": inbox or [],
        "events": events or [],
        "metadata": {"n_patients": len(pats), "env_chars": env_chars, "env_tokens_approx": env_chars // 4},
    }


def _provider_near_mapped(p: dict, day: str) -> str:
    # Encounter providers are already mapped to short IDs at this point.
    return _provider_near(p, day)


# ---------------------------------------------------------------- families
def build_update(pool, n, level, rng, task_id, max_obs, specialties):
    pats = select_patients(pool, n, rng, max_obs)
    flags = _split(n, 0.5, rng)
    for p, high in zip(pats, flags):
        p["medications"] = [m for m in p["medications"] if ingredient(m["name"]) != "lisinopril"]
        # Drop every potassium measurement (Synthea has two potassium tests) so only ours remain.
        p["observations"] = [o for o in p["observations"] if "potassium" not in o["name"].lower()]
        start = days_before(rng.randint(60, 1500))
        p["medications"].append({"id": "", "name": "lisinopril 10 MG Oral Tablet", "start": start, "stop": "",
                                 "prescriber": _provider_near(p, start), "reason": "Hypertension"})

        def val(hi: bool) -> str:
            return f"{rng.uniform(5.2, 6.2):.1f}" if hi else f"{rng.uniform(3.6, 4.7):.1f}"

        last_day = rng.randint(1, 20)
        values = [(days_before(last_day), val(high))]
        if level >= 1:
            gaps = sorted(rng.sample(range(40, 700), 3))
            values += [(days_before(last_day + g), val(not high)) for g in gaps]
        for day, v in values:
            p["observations"].append({"date": day, "name": POTASSIUM, "value": v, "units": "mmol/L"})

    events = []
    if level == 2:
        flip = [rng.choice([p for p, h in zip(pats, flags) if h]), rng.choice([p for p, h in zip(pats, flags) if not h])]
        for p in flip:
            was_high = float(max((o for o in p["observations"] if o["name"] == POTASSIUM),
                                 key=lambda o: o["date"])["value"]) > 5.0
            new = f"{rng.uniform(3.6, 4.7):.1f}" if was_high else f"{rng.uniform(5.2, 6.2):.1f}"
            events.append({
                "id": f"EV-{p['id']}",
                "trigger": {"type": "after_read", "tool": "get_labs", "patient": p["id"], "match": "potassium", "delay": 2},
                "effect": {"type": "add_observation", "patient": p["id"],
                           "observation": {"date": TODAY, "name": POTASSIUM, "value": new, "units": "mmol/L"}},
                "alert": f"[ALERT] New lab result posted for {p['id']}: Potassium.",
            })
    return finalize(pats, "update", level, task_id, events=events, provider_specialty=specialties)


DISTRACTOR_RANGES = {
    # level: (a1c pass, a1c fail, days pass, days fail)
    0: ((7.5, 9.0), (5.0, 5.6), (270, 400), (20, 60)),
    1: ((6.7, 7.2), (6.0, 6.3), (200, 240), (130, 160)),
    2: ((6.6, 6.7), (6.4, 6.5), (183, 188), (176, 182)),
}


def build_distractor(pool, n, level, rng, task_id, max_obs, specialties):
    pats = select_patients(pool, n, rng, max_obs)
    a_pass, a_fail, d_pass, d_fail = DISTRACTOR_RANGES[level]
    kinds = ["qualify"] * max(1, round(n * 0.4))
    rest = n - len(kinds)
    kinds += ["fail_a1c"] * (rest // 2) + ["fail_visit"] * (rest - rest // 2)
    rng.shuffle(kinds)
    for p, kind in zip(pats, kinds):
        days = rng.randint(*(d_fail if kind == "fail_visit" else d_pass))
        lo, hi = a_fail if kind == "fail_a1c" else a_pass
        v = round(rng.uniform(lo, hi), 1)
        last = days_before(days)
        provider = _provider_near(p, last)
        _truncate_after(p, last)
        p["observations"] = [o for o in p["observations"] if o["name"] != A1C]
        _new_encounter(p, last, "wellness", "General examination of patient (procedure)", provider=provider)
        prior = round(v + rng.uniform(0, 0.3), 1) if kind != "fail_a1c" else round(v - rng.uniform(0, 0.3), 1)
        for day, val in [(days_before(365, last), prior), (last, v)]:
            p["observations"].append({"date": day, "name": A1C, "value": f"{val:.1f}", "units": "%"})
        p["_kind"] = kind
    task = finalize(pats, "distractor", level, task_id, provider_specialty=specialties)
    for p in task["patients"].values():
        p.pop("_kind", None)
    return task


def build_distance(pool, n, level, rng, task_id, max_obs, specialties):
    pats = select_patients(pool, n, rng, max_obs)
    flags = _split(n, 0.4, rng)
    for p, allergic in zip(pats, flags):
        p["allergies"] = [a for a in p["allergies"] if "penicillin" not in a["substance"].lower()]
        for e in p["encounters"]:
            if rng.random() < 0.25:
                e["note"] = rng.choice(FILLER_NOTES)
        lo, hi = (2920, 5475) if level == 2 else (180, 540)
        day = days_before(rng.randint(lo, hi))
        note = ALLERGY_NOTE if (allergic and level >= 1) else rng.choice(FILLER_NOTES)
        _new_encounter(p, day, "ambulatory", "Encounter for problem", note=note)
        if allergic and level == 0:
            p["allergies"].append({"date": day, "substance": "Penicillin V", "reaction": "Wheal (finding)"})
        _new_encounter(p, TODAY, "ambulatory", "Encounter for symptom", reason="Streptococcal sore throat (disorder)")
        p["conditions"].append({"date": TODAY, "description": "Streptococcal sore throat (disorder)", "resolved": ""})
    return finalize(pats, "distance", level, task_id, provider_specialty=specialties)


def build_interleave(pool, n, level, rng, task_id, max_obs, specialties):
    if n < 5:
        raise ValueError("interleave needs at least 5 patients")
    pats = select_patients(pool, n, rng, max_obs)
    targets = rng.sample(pats, 5)
    kinds = ["schedule", "schedule", "hold", "order", rng.choice(["schedule", "hold", "order"])]
    rng.shuffle(kinds)
    inbox = []
    for i, (p, kind) in enumerate(zip(targets, kinds), start=1):
        sender = rng.choice([e["provider"] for e in p["encounters"] if e["provider"]])
        if kind == "schedule":
            typ = rng.choice(APPOINTMENT_TYPES)
            art = "an" if typ[0].lower() in "aeiou" else "a"
            key, text = ["schedule", p["id"], typ], f"Please schedule {art} {typ} for patient {p['id']}."
        elif kind == "hold":
            active = [m for m in p["medications"] if not m["stop"] and ingredient(m["name"])]
            if not active:
                m = {"id": "", "name": "lisinopril 10 MG Oral Tablet", "start": days_before(400), "stop": "",
                     "prescriber": _provider_near(p, days_before(400)), "reason": "Hypertension"}
                p["medications"].append(m)
                active = [m]
            ing = ingredient(rng.choice(active)["name"])
            key, text = ["hold", p["id"], ing], f"Please hold {ing} for patient {p['id']}."
        else:
            drug = rng.choice(FORMULARY)
            key, text = ["order", p["id"], ingredient(drug)], f"Please order {drug} for patient {p['id']}."
        inbox.append({"id": f"MSG{i}", "date": TODAY, "from": sender, "text": text, "delivered": True,
                      "_action": {"op": "request", "key": key}})

    events = []
    if level >= 1:
        sched = [m for m in inbox if m["_action"]["key"][0] == "schedule"]
        other = [m for m in inbox if m["_action"]["key"][0] != "schedule"]
        a, b = rng.choice(sched), rng.choice(other)
        old = a["_action"]["key"][2]
        new = rng.choice([t for t in APPOINTMENT_TYPES if t != old])
        pa, pb = a["_action"]["key"][1], b["_action"]["key"][1]
        inbox.append({"id": "MSG6", "date": TODAY, "from": a["from"], "delivered": level == 1,
                      "text": f"Update to my message {a['id']}: for patient {pa}, please book "
                              f"{'an' if new[0].lower() in 'aeiou' else 'a'} {new} instead of the {old}.",
                      "_action": {"op": "replace", "target": a["id"], "key": ["schedule", pa, new]}})
        inbox.append({"id": "MSG7", "date": TODAY, "from": b["from"], "delivered": level == 1,
                      "text": f"Please disregard my message {b['id']} about patient {pb}; that request is no longer needed.",
                      "_action": {"op": "cancel", "target": b["id"]}})
        if level == 2:
            events.append({"id": "EV-inbox", "trigger": {"type": "after_writes", "count": 3},
                           "effect": {"type": "deliver_messages", "ids": ["MSG6", "MSG7"]},
                           "alert": "[ALERT] 2 new messages arrived in the inbox."})
    return finalize(pats, "interleave", level, task_id, inbox=inbox, events=events, provider_specialty=specialties)


BUILDERS = {
    "update": build_update,
    "distractor": build_distractor,
    "distance": build_distance,
    "interleave": build_interleave,
}
