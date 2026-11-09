"""Generate mock-EHR tasks from Synthea.

Writes one JSON file per task to <out>/tasks/ and an index to <out>/index.jsonl.

Usage:
    python -m ehr.generate --synthea-dir data/raw/synthea/csv --out data/ehr \
        --families update distractor distance interleave --levels 0 1 2 \
        --n-tasks 5 --n-patients 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from ehr.env import EHREnv
from ehr.families import BUILDERS, FAMILIES
from ehr.grade import expected_actions
from ehr.records import load_providers, load_records


def generate(pool, specialties, families, levels, n_tasks, n_patients, seed, max_obs) -> list[dict]:
    tasks = []
    for fam in families:
        for level in levels:
            for k in range(n_tasks):
                # The same seed and index pick the same patients at every level, so levels
                # differ only in the difficulty factor, not in which charts are used.
                rng = random.Random(f"{seed}-{fam}-{k}")
                tid = f"{fam}-L{level}-{k:03d}"
                task = BUILDERS[fam](pool, n_patients, level, rng, tid, max_obs, specialties)
                task["metadata"]["n_expected_initial"] = len(expected_actions(EHREnv(task)))
                task["metadata"]["seed"] = seed
                tasks.append(task)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthea-dir", default="data/raw/synthea/csv")
    ap.add_argument("--out", default="data/ehr")
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-tasks", type=int, default=5, help="tasks per (family, level)")
    ap.add_argument("--n-patients", type=int, default=8)
    ap.add_argument("--max-obs", type=int, default=150, help="most recent lab results kept per patient")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading Synthea records...")
    pool = load_records(args.synthea_dir)
    specialties = load_providers(args.synthea_dir)
    print(f"pool: {len(pool)} living adult patients")

    tasks = generate(pool, specialties, args.families, args.levels, args.n_tasks, args.n_patients,
                     args.seed, args.max_obs)
    out = Path(args.out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    with (out / "index.jsonl").open("w") as idx:
        for t in tasks:
            (out / "tasks" / f"{t['task_id']}.json").write_text(json.dumps(t))
            idx.write(json.dumps({"task_id": t["task_id"], "family": t["family"], "level": t["level"],
                                  **t["metadata"]}) + "\n")

    sizes = defaultdict(list)
    for t in tasks:
        sizes[(t["family"], t["level"])].append(t["metadata"]["env_tokens_approx"])
    for (fam, lvl), v in sorted(sizes.items()):
        print(f"{fam:11s} L{lvl}: {len(v)} tasks, environment ~{int(statistics.median(v)):,} tokens (median)")
    print(f"wrote {len(tasks)} tasks to {out / 'tasks'}")


if __name__ == "__main__":
    main()
