"""Generate scalable multi-patient questions from Synthea charts.

Each task samples a collection of patient charts and asks which patients
satisfy a chain of conditions. Condition kinds:

- diagnosis: has (or has no) recorded diagnosis with an exact name
- medication: has (or has never) been prescribed a medication containing an ingredient
- lab: most recent recorded value of a lab or vital is above (or not above) a threshold
- sex: is female or male
- birth: born before (or in or after) a year
- deceased: has (or has no) recorded death date

Lab conditions need the latest value across the whole chart, so the agent
has to keep track of details spread over many encounters.

Usage:
    python -m testbeds.biomedical.generate --synthea-dir data/raw/synthea/csv \
        --out data/biomedical --n-docs 10 50 200 --n-steps 1 3 5 --n-tasks 10
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.chain import default_compatible, generate_grid, make_chain_task
from core.docstore import DocStore
from core.task import Task, save_tasks
from testbeds.biomedical.synthea import TRACKED_OBSERVATIONS, Patient, load_synthea

# Clinical diagnoses common enough to split collections (about 5% to 65% of
# patients). Names match the chart text exactly. Social and behavioral
# findings are left out on purpose.
DIAGNOSES = [
    "Viral sinusitis (disorder)",
    "Acute viral pharyngitis (disorder)",
    "Body mass index 30+ - obesity (finding)",
    "Acute bronchitis (disorder)",
    "Prediabetes",
    "Anemia (disorder)",
    "Hypertension",
    "Normal pregnancy",
    "Chronic sinusitis (disorder)",
    "Chronic low back pain (finding)",
    "Otitis media",
    "Streptococcal sore throat (disorder)",
    "Hyperlipidemia",
    "Chronic neck pain (finding)",
    "Sprain of ankle",
    "COVID-19",
    "Polyp of colon",
    "Diabetes",
    "Coronary Heart Disease",
    "Metabolic syndrome X (disorder)",
]

INGREDIENTS = [
    "acetaminophen", "ibuprofen", "naproxen", "amoxicillin", "lisinopril",
    "hydrochlorothiazide", "amlodipine", "simvastatin", "penicillin",
    "epinephrine", "clopidogrel", "nitroglycerin", "fluticasone",
]

# code -> threshold
LAB_THRESHOLDS = {
    "39156-5": 30.0,  # BMI
    "8480-6": 130.0,  # systolic blood pressure
    "4548-4": 6.5,  # hemoglobin A1c
    "2093-3": 200.0,  # total cholesterol
    "18262-6": 100.0,  # LDL cholesterol
}

BIRTH_YEARS = ["1950", "1970", "1990"]


@dataclass(frozen=True)
class BioCondition:
    kind: str
    key: str
    positive: bool

    def holds(self, p: Patient) -> bool:
        if self.kind == "diagnosis":
            val = self.key in p.diagnoses
        elif self.kind == "medication":
            val = p.takes(self.key)
        elif self.kind == "lab":
            v = p.latest_obs.get(self.key)
            val = v is not None and v > LAB_THRESHOLDS[self.key]
        elif self.kind == "sex":
            val = p.sex == self.key
        elif self.kind == "birth":
            val = p.birth_year < int(self.key)
        elif self.kind == "deceased":
            val = p.deceased
        else:
            raise ValueError(self.kind)
        return val if self.positive else not val

    def phrase(self) -> str:
        k, pos = self.key, self.positive
        if self.kind == "diagnosis":
            return f'have a recorded diagnosis of "{k}"' if pos else f'have no recorded diagnosis of "{k}"'
        if self.kind == "medication":
            return (
                f"have been prescribed a medication containing {k}"
                if pos
                else f"have never been prescribed a medication containing {k}"
            )
        if self.kind == "lab":
            name, unit = TRACKED_OBSERVATIONS[k]
            thr = f"{LAB_THRESHOLDS[k]:g} {unit}"
            if pos:
                return f"have a most recent recorded {name} above {thr}"
            return f"do not have a most recent recorded {name} above {thr} (including patients with no {name} recorded)"
        if self.kind == "sex":
            return "are female" if k == "F" else "are male"
        if self.kind == "birth":
            return f"were born before {k}" if pos else f"were born in or after {k}"
        if self.kind == "deceased":
            return "have a recorded death date" if pos else "have no recorded death date"
        raise ValueError(self.kind)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "key": self.key, "positive": self.positive}


def all_conditions() -> list[BioCondition]:
    conds = []
    for d in DIAGNOSES:
        conds += [BioCondition("diagnosis", d, True), BioCondition("diagnosis", d, False)]
    for i in INGREDIENTS:
        conds += [BioCondition("medication", i, True), BioCondition("medication", i, False)]
    for code in LAB_THRESHOLDS:
        conds += [BioCondition("lab", code, True), BioCondition("lab", code, False)]
    conds += [BioCondition("sex", "F", True), BioCondition("sex", "M", True)]
    for y in BIRTH_YEARS:
        conds += [BioCondition("birth", y, True), BioCondition("birth", y, False)]
    conds += [BioCondition("deceased", "deceased", True), BioCondition("deceased", "deceased", False)]
    return conds


def bio_compatible(chosen: list[BioCondition], cond: BioCondition) -> bool:
    if not default_compatible(chosen, cond):
        return False
    # At most one sex, birth-year, and death condition per chain.
    if cond.kind in {"sex", "birth", "deceased"} and any(c.kind == cond.kind for c in chosen):
        return False
    return True


def build_instructions(n_docs: int, conds: list[BioCondition]) -> str:
    lines = [
        f"You have a collection of {n_docs} synthetic patient charts in the docs/ folder.",
        "Each file is named by its patient ID (for example P0042.txt).",
        "Answer each question with the IDs of every patient that satisfies it.",
        "End your answer with a line of the form: ANSWER: P0001, P0002",
        "If no patient qualifies, write: ANSWER: none",
    ]
    notes = []
    kinds = {c.kind for c in conds}
    if "diagnosis" in kinds:
        notes.append("- A diagnosis counts if it appears anywhere in the chart, even if later resolved. Names must match exactly.")
    if "medication" in kinds:
        notes.append("- A medication counts if any prescription in the chart lists the ingredient, including combination products.")
    if "lab" in kinds:
        notes.append("- The most recent recorded value is the value on the latest date that measurement appears in the chart.")
    if notes:
        lines += ["", "Notes:"] + notes
    return "\n".join(lines)


def make_task(pool: list[Patient], n_docs: int, n_steps: int, rng: random.Random, task_id: str) -> Task | None:
    return make_chain_task(
        pool,
        all_conditions(),
        n_docs,
        n_steps,
        rng,
        task_id=task_id,
        domain="biomedical",
        noun="patients",
        build_instructions=build_instructions,
        compatible=bio_compatible,
        kind_boost={"lab": 0.25, "medication": 0.2},
        metadata={"source": "Synthea sample data (csv, Nov 2021)"},
    )


def build_pool(patients: list[Patient], max_doc_chars: int) -> list[Patient]:
    """Drop patients whose lab values are ambiguous or whose charts are extreme outliers."""
    return [p for p in patients if not p.ambiguous_obs and len(p.text) <= max_doc_chars]


def generate(patients, n_docs_list, n_steps_list, n_tasks, seed, max_doc_chars=400_000) -> list[Task]:
    pool = build_pool(patients, max_doc_chars)
    print(f"pool: {len(pool)} of {len(patients)} patients")
    return generate_grid(
        lambda nd, ns, rng, tid: make_task(pool, nd, ns, rng, tid),
        "bio",
        n_docs_list,
        n_steps_list,
        n_tasks,
        seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthea-dir", default="data/raw/synthea/csv", help="folder with the Synthea CSV tables")
    ap.add_argument("--out", default="data/biomedical")
    ap.add_argument("--n-docs", type=int, nargs="+", default=[10, 50, 200])
    ap.add_argument("--n-steps", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--n-tasks", type=int, default=10, help="tasks per (n_docs, n_steps) cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-doc-chars", type=int, default=400_000, help="skip charts longer than this")
    args = ap.parse_args()

    patients = load_synthea(args.synthea_dir)
    out = Path(args.out)
    store = DocStore(out)
    for p in patients:
        store.write(p.doc_id, p.text)
    (out / "patients.json").write_text(
        json.dumps([{"doc_id": p.doc_id, "source_id": p.source_id} for p in patients], indent=1)
    )

    tasks = generate(patients, args.n_docs, args.n_steps, args.n_tasks, args.seed, args.max_doc_chars)
    save_tasks(tasks, out / "tasks.jsonl")
    kinds = Counter(s.condition["kind"] for t in tasks for s in t.steps)
    print(f"wrote {len(tasks)} tasks to {out / 'tasks.jsonl'} ({dict(kinds)} conditions)")


if __name__ == "__main__":
    main()
