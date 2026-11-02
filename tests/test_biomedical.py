import csv
import random

from testbeds.biomedical.generate import BioCondition, all_conditions, make_task
from testbeds.biomedical.synthea import load_synthea
from testbeds.biomedical.verify import holds_from_text

TABLES = {
    "patients.csv": ["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "FIRST", "LAST", "SSN"],
    "encounters.csv": ["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS", "DESCRIPTION", "REASONDESCRIPTION"],
    "conditions.csv": ["START", "STOP", "PATIENT", "ENCOUNTER", "DESCRIPTION"],
    "medications.csv": ["START", "STOP", "PATIENT", "ENCOUNTER", "DESCRIPTION", "REASONDESCRIPTION"],
    "observations.csv": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "VALUE", "UNITS"],
    "procedures.csv": ["START", "PATIENT", "ENCOUNTER", "DESCRIPTION", "REASONDESCRIPTION"],
    "immunizations.csv": ["DATE", "PATIENT", "ENCOUNTER", "DESCRIPTION"],
    "allergies.csv": ["START", "PATIENT", "ENCOUNTER", "DESCRIPTION"],
    "careplans.csv": ["START", "PATIENT", "ENCOUNTER", "DESCRIPTION", "REASONDESCRIPTION"],
}


def write_tables(d, rows):
    for name, cols in TABLES.items():
        with (d / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows.get(name, []):
                w.writerow({c: r.get(c, "") for c in cols})


def small_dataset(tmp_path):
    rows = {
        "patients.csv": [
            {"Id": "a", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2010-05-05", "GENDER": "F", "FIRST": "Ann", "SSN": "999-11-2222"},
            {"Id": "b", "BIRTHDATE": "1995-06-01", "GENDER": "M", "FIRST": "Bob"},
        ],
        "encounters.csv": [
            {"Id": "e1", "START": "2000-01-01T10:00:00Z", "PATIENT": "a", "ENCOUNTERCLASS": "wellness", "DESCRIPTION": "Checkup"},
            {"Id": "e2", "START": "2005-01-01T10:00:00Z", "PATIENT": "a", "ENCOUNTERCLASS": "wellness", "DESCRIPTION": "Checkup"},
            {"Id": "e3", "START": "2015-01-01T10:00:00Z", "PATIENT": "b", "ENCOUNTERCLASS": "urgentcare", "DESCRIPTION": "Visit"},
        ],
        "conditions.csv": [
            {"START": "2000-01-01", "PATIENT": "a", "ENCOUNTER": "e1", "DESCRIPTION": "Hypertension"},
            {"START": "2015-01-01", "STOP": "2015-02-01", "PATIENT": "b", "ENCOUNTER": "e3", "DESCRIPTION": "Otitis media"},
        ],
        "medications.csv": [
            {"START": "2000-01-01T10:00:00Z", "PATIENT": "a", "ENCOUNTER": "e1", "DESCRIPTION": "lisinopril 10 MG Oral Tablet", "REASONDESCRIPTION": "Hypertension"},
            {"START": "2015-01-01T10:00:00Z", "PATIENT": "b", "ENCOUNTER": "e3", "DESCRIPTION": "Amoxicillin 250 MG Oral Capsule"},
        ],
        "observations.csv": [
            {"DATE": "2000-01-01T10:00:00Z", "PATIENT": "a", "ENCOUNTER": "e1", "CODE": "39156-5", "DESCRIPTION": "Body Mass Index", "VALUE": "35.0", "UNITS": "kg/m2"},
            {"DATE": "2005-01-01T10:00:00Z", "PATIENT": "a", "ENCOUNTER": "e2", "CODE": "39156-5", "DESCRIPTION": "Body Mass Index", "VALUE": "28.0", "UNITS": "kg/m2"},
            {"DATE": "2015-01-01T10:00:00Z", "PATIENT": "b", "ENCOUNTER": "e3", "CODE": "8480-6", "DESCRIPTION": "Systolic Blood Pressure", "VALUE": "140", "UNITS": "mm[Hg]"},
            {"DATE": "2015-01-01T11:00:00Z", "PATIENT": "b", "ENCOUNTER": "e3", "CODE": "8480-6", "DESCRIPTION": "Systolic Blood Pressure", "VALUE": "120", "UNITS": "mm[Hg]"},
        ],
        "immunizations.csv": [
            {"DATE": "2003-01-01T10:00:00Z", "PATIENT": "a", "ENCOUNTER": "missing", "DESCRIPTION": "Influenza vaccine"},
        ],
    }
    write_tables(tmp_path, rows)
    return load_synthea(tmp_path)


def test_render_and_facts(tmp_path):
    a, b = small_dataset(tmp_path)
    assert (a.doc_id, b.doc_id) == ("P0001", "P0002")
    assert "Death date: 2010-05-05" in a.text and "Death date: none (living)" in b.text
    assert "Ann" not in a.text and "999-11-2222" not in a.text  # identifiers left out
    assert "Medication: lisinopril 10 MG Oral Tablet for Hypertension, started 2000-01-01" in a.text
    assert "Diagnosis: Otitis media (resolved 2015-02-01)" in b.text
    assert "2003-01-01 Immunization: Influenza vaccine" in a.text  # record with no encounter
    assert a.latest_obs["39156-5"] == 28.0  # latest, not highest
    assert "8480-6" in b.ambiguous_obs  # two values on the same day


def test_conditions_match_text(tmp_path):
    patients = small_dataset(tmp_path)
    for p in patients:
        for cond in all_conditions():
            if cond.kind == "lab" and cond.key in p.ambiguous_obs:
                continue
            assert cond.holds(p) == holds_from_text(p.text, cond.to_dict()), (p.doc_id, cond)


def test_specific_conditions(tmp_path):
    a, b = small_dataset(tmp_path)
    assert BioCondition("lab", "39156-5", True).holds(a) is False  # latest BMI 28
    assert BioCondition("medication", "amoxicillin", True).holds(b)
    assert BioCondition("diagnosis", "Hypertension", False).holds(b)
    assert BioCondition("birth", "1950", True).holds(a)
    assert BioCondition("deceased", "deceased", True).holds(a)
    assert BioCondition("sex", "M", True).holds(b)


def test_make_task_on_real_shaped_pool(tmp_path):
    rng = random.Random(0)
    rows = {"patients.csv": [], "encounters.csv": [], "conditions.csv": [], "observations.csv": []}
    for i in range(40):
        pid = f"p{i}"
        rows["patients.csv"].append({"Id": pid, "BIRTHDATE": f"{rng.randint(1930, 2005)}-01-01", "GENDER": rng.choice("FM")})
        rows["encounters.csv"].append({"Id": f"e{i}", "START": "2010-01-01T00:00:00Z", "PATIENT": pid, "ENCOUNTERCLASS": "wellness", "DESCRIPTION": "Checkup"})
        for dx in ["Hypertension", "Prediabetes", "Anemia (disorder)"]:
            if rng.random() < 0.5:
                rows["conditions.csv"].append({"START": "2010-01-01", "PATIENT": pid, "ENCOUNTER": f"e{i}", "DESCRIPTION": dx})
        rows["observations.csv"].append({"DATE": "2010-01-01T00:00:00Z", "PATIENT": pid, "ENCOUNTER": f"e{i}", "CODE": "39156-5", "DESCRIPTION": "Body Mass Index", "VALUE": str(rng.uniform(20, 40)), "UNITS": "kg/m2"})
    write_tables(tmp_path, rows)
    pool = load_synthea(tmp_path)
    by_id = {p.doc_id: p for p in pool}
    t = make_task(pool, 20, 3, random.Random(1), "t")
    assert t is not None and t.n_steps == 3
    prev = set(t.doc_ids)
    for s in t.steps:
        assert set(s.gold) < prev and s.gold
        for d in prev:
            assert holds_from_text(by_id[d].text, s.condition) == (d in s.gold)
        prev = set(s.gold)
