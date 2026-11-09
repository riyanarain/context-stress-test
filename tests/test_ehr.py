import random

import pytest

from ehr.env import EHREnv
from ehr.families import BUILDERS, FAMILIES
from ehr.grade import grade
from ehr.oracle import OracleAgent
from ehr.records import TODAY, days_before


def fake_pool(n=14, seed=0):
    rng = random.Random(seed)
    pool = []
    for i in range(n):
        encs, obs = [], []
        for k in range(30):
            day = days_before(rng.randint(30, 6000))
            encs.append({"id": "", "date": day, "class": "wellness", "description": "Checkup",
                         "provider": f"prov-{rng.randint(1, 4)}", "reason": "", "note": ""})
            obs.append({"date": day, "name": "Body Mass Index", "value": f"{rng.uniform(20, 35):.1f}", "units": "kg/m2"})
            obs.append({"date": day, "name": "Potassium", "value": "4.1", "units": "mmol/L"})
        pool.append({
            "id": f"P{i + 1:04d}", "sex": rng.choice("FM"), "birth_date": "1960-05-05",
            "encounters": sorted(encs, key=lambda e: e["date"]),
            "conditions": [{"date": days_before(900), "description": "Hypertension", "resolved": ""}],
            "medications": [{"id": "", "name": "amLODIPine 2.5 MG Oral Tablet", "start": days_before(800), "stop": "",
                             "prescriber": "prov-1", "reason": "Hypertension"}],
            "observations": sorted(obs, key=lambda o: o["date"]),
            "allergies": [],
        })
    return pool


def build(family, level, k=0, n=8):
    return BUILDERS[family](fake_pool(), n, level, random.Random(f"x-{family}-{k}"), f"{family}-L{level}", 200, {})


def run_oracle(task):
    env = EHREnv(task)
    agent = OracleAgent(env)
    for _ in range(300):
        env.call(*agent.next_call())
        if env.done:
            break
    return env


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", [0, 1, 2])
def test_oracle_solves_every_family_and_level(family, level):
    for k in range(3):
        env = run_oracle(build(family, level, k))
        g = grade(env)
        assert g["exact"] == 1.0 and g["finished"], (family, level, g)
        assert g["events_fired"] == g["events_total"]


def test_levels_keep_environment_size():
    for fam in FAMILIES:
        sizes = [build(fam, lvl)["metadata"]["env_chars"] for lvl in (0, 1, 2)]
        assert max(sizes) / min(sizes) < 1.1, (fam, sizes)


def test_tools_and_errors():
    task = build("update", 1)
    env = EHREnv(task)
    pid = sorted(env.patients)[0]
    assert "Potassium" in env.call("get_labs", {"patient_id": pid, "test_name": "potassium"})
    assert env.call("get_labs", {"patient_id": "NOPE"}).startswith("Error")
    assert env.call("hold_medication", {"patient_id": pid, "medication": "warfarin"}).startswith("Error")
    out = env.call("hold_medication", {"patient_id": pid, "medication": "lisinopril"})
    aid = out.split("Action ID ")[1].rstrip(".")
    assert ("hold", pid, "lisinopril") in env.active_actions()
    assert "ON HOLD" in env.call("get_medications", {"patient_id": pid})
    env.call("cancel_action", {"action_id": aid})
    assert env.active_actions() == set()
    assert env.call("schedule_appointment", {"patient_id": pid, "appointment_type": "Spa day"}).startswith("Error")
    assert env.call("bogus_tool", {}).startswith("Error")
    assert env.call("get_patient", "{not json").startswith("Error")


def test_update_event_fires_after_read_and_changes_gold():
    task = build("update", 2)
    env = EHREnv(task)
    ev = task["events"][0]
    pid = ev["trigger"]["patient"]
    before = grade(env)["n_gold"]
    env.call("get_labs", {"patient_id": pid, "test_name": "glucose"})  # wrong test: no trigger
    assert not env.events[0]["fired"]
    env.call("get_labs", {"patient_id": pid, "test_name": "potassium"})
    out1 = env.call("list_patients", {})
    assert "[ALERT]" not in out1
    out2 = env.call("list_patients", {})
    assert "[ALERT] New lab result posted for " + pid in out2
    assert env.events[0]["fired"]
    assert grade(env)["n_gold"] != before  # one patient flipped


def test_interleave_amendments_arrive_after_three_writes():
    task = build("interleave", 2)
    env = EHREnv(task)
    first = env.call("read_inbox", {})
    assert "MSG6" not in first
    for m in task["inbox"][:3]:
        kind, pid, val = m["_action"]["key"]
        if kind == "schedule":
            out = env.call("schedule_appointment", {"patient_id": pid, "appointment_type": val})
        elif kind == "hold":
            out = env.call("hold_medication", {"patient_id": pid, "medication": val})
        else:
            out = env.call("order_medication", {"patient_id": pid, "medication": val})
    assert "[ALERT] 2 new messages" in out
    later = env.call("read_inbox", {})
    assert "NEW  MSG6" in later and "NEW  MSG1" not in later


def test_distance_allergy_placement():
    for level in (0, 1, 2):
        task = build("distance", level)
        allergic = [p for p in task["patients"].values()
                    if any("penicillin" in a["substance"].lower() for a in p["allergies"])
                    or any("penicillin" in e["note"].lower() for e in p["encounters"])]
        assert allergic
        for p in allergic:
            in_list = any("penicillin" in a["substance"].lower() for a in p["allergies"])
            note = [e for e in p["encounters"] if "penicillin" in e["note"].lower()]
            if level == 0:
                assert in_list and not note
            else:
                assert not in_list and len(note) == 1
                age_days = (int(TODAY[:4]) - int(note[0]["date"][:4])) * 365
                assert (age_days >= 7 * 365) == (level == 2)


def test_distractor_level2_has_near_misses():
    task = build("distractor", 2, n=10)
    vals = []
    for p in task["patients"].values():
        a1c = max((o for o in p["observations"] if o["name"].startswith("Hemoglobin A1c")), key=lambda o: o["date"])
        vals.append(float(a1c["value"]))
    assert all(6.4 <= v <= 6.7 for v in vals)
