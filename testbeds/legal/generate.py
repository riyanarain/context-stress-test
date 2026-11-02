"""Generate scalable multi-contract questions from CUAD.

Each task samples a collection of contracts and asks which ones satisfy a
chain of conditions (clause present or absent, governing law is or is not X).
Two knobs:

- n_docs: collection size (data size)
- n_steps: number of chained conditions (horizon). In multi-turn mode each
  step narrows the previous answer, so a mistake early on carries forward.

Every step must strictly shrink the candidate set and leave at least one
contract, so no step is redundant and no answer is empty. Gold answers come
from the CUAD lawyer annotations.

Usage:
    python -m testbeds.legal.generate --cuad-dir data/raw/cuad --out data/legal \
        --n-docs 10 50 200 --n-steps 1 3 5 --n-tasks 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.docstore import DocStore
from core.task import Step, Task, save_tasks
from testbeds.legal.cuad import Contract, load_category_descriptions, load_cuad

# Presence categories used as conditions, with readable names. These all appear
# in at least ~60 of the 510 contracts, so conditions split collections well.
CLAUSE_LABELS = {
    "Anti-Assignment": "an anti-assignment clause",
    "Cap On Liability": "a cap on liability",
    "License Grant": "a license grant",
    "Audit Rights": "audit rights",
    "Termination For Convenience": "a termination for convenience clause",
    "Post-Termination Services": "a post-termination services clause",
    "Exclusivity": "an exclusivity clause",
    "Renewal Term": "a renewal term",
    "Insurance": "an insurance requirement",
    "Revenue/Profit Sharing": "a revenue or profit sharing clause",
    "Minimum Commitment": "a minimum commitment",
    "Non-Transferable License": "a non-transferable license",
    "Ip Ownership Assignment": "an IP ownership assignment",
    "Change Of Control": "a change of control clause",
    "Non-Compete": "a non-compete clause",
    "Uncapped Liability": "an uncapped liability clause",
    "Covenant Not To Sue": "a covenant not to sue",
    "Rofr/Rofo/Rofn": "a right of first refusal, offer, or negotiation",
    "Volume Restriction": "a volume restriction",
    "Liquidated Damages": "a liquidated damages clause",
    "No-Solicit Of Employees": "a no-solicit of employees clause",
}

# Jurisdictions common enough to be useful conditions.
JURISDICTIONS = ["New York", "California", "Delaware", "Texas", "Florida", "England and Wales"]


@dataclass(frozen=True)
class Condition:
    kind: str  # "clause" or "law"
    key: str  # CUAD category or jurisdiction
    positive: bool

    def holds(self, c: Contract) -> bool:
        if self.kind == "clause":
            val = c.has(self.key)
        else:
            val = c.governing_law == self.key
        return val if self.positive else not val

    def phrase(self) -> str:
        if self.kind == "clause":
            label = CLAUSE_LABELS[self.key]
            return f"contain {label}" if self.positive else f"do not contain {label}"
        if self.positive:
            return f"are governed by the law of {self.key}"
        return f"are not governed by the law of {self.key}"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "key": self.key, "positive": self.positive}


def all_conditions() -> list[Condition]:
    conds = []
    for cat in CLAUSE_LABELS:
        conds += [Condition("clause", cat, True), Condition("clause", cat, False)]
    for j in JURISDICTIONS:
        conds += [Condition("law", j, True), Condition("law", j, False)]
    return conds


def build_instructions(n_docs: int, conds: list[Condition], descriptions: dict[str, str]) -> str:
    lines = [
        f"You have a collection of {n_docs} commercial contracts in the docs/ folder.",
        "Each file is named by its contract ID (for example C0042.txt).",
        "Answer each question with the IDs of every contract that satisfies it.",
        "End your answer with a line of the form: ANSWER: C0001, C0002",
        "If no contract qualifies, write: ANSWER: none",
    ]
    defs = []
    for c in conds:
        if c.kind == "clause":
            d = descriptions.get(c.key.lower())
            if d:
                defs.append(f"- {c.key}: {d}")
    if any(c.kind == "law" for c in conds):
        defs.append(
            "- Governing law: the jurisdiction whose law the contract says governs it. "
            "A contract with no governing law clause is not governed by any listed jurisdiction."
        )
    if defs:
        lines += ["", "Definitions used for these questions:"] + defs
    return "\n".join(lines)


def make_task(
    pool: list[Contract],
    n_docs: int,
    n_steps: int,
    rng: random.Random,
    task_id: str,
    descriptions: dict[str, str],
    max_attempts: int = 500,
) -> Task | None:
    conds_all = all_conditions()
    for _ in range(max_attempts):
        docs = rng.sample(pool, n_docs)
        by_id = {c.doc_id: c for c in docs}
        current = set(by_id)
        chosen: list[Condition] = []
        golds: list[list[str]] = []
        for s in range(n_steps):
            remaining = n_steps - s
            used = {(c.kind, c.key) for c in chosen}
            options = []
            for cond in conds_all:
                if (cond.kind, cond.key) in used:
                    continue
                if cond.kind == "law" and cond.positive and any(
                    c.kind == "law" and c.positive for c in chosen
                ):
                    continue
                new = {d for d in current if cond.holds(by_id[d])}
                # Strictly shrink, and leave room for the remaining steps to shrink too.
                if len(new) >= remaining and len(new) < len(current):
                    options.append((cond, new))
            if not options:
                break
            # Prefer conditions that keep a moderate fraction, so chains stay long.
            good = [(c, n) for c, n in options if 0.3 <= len(n) / len(current) <= 0.85] or options
            # Clause conditions outnumber law conditions about 3 to 1, so give
            # law conditions a fixed share to keep both kinds in the mix.
            law = [(c, n) for c, n in good if c.kind == "law"]
            if law and rng.random() < 0.3:
                good = law
            cond, new = rng.choice(good)
            chosen.append(cond)
            current = new
            golds.append(sorted(new))
        if len(chosen) != n_steps:
            continue

        steps = []
        for i, (cond, gold) in enumerate(zip(chosen, golds)):
            if i == 0:
                q = f"Which contracts in the collection {cond.phrase()}?"
            else:
                q = f"Of the contracts in your previous answer, which {cond.phrase()}?"
            steps.append(Step(question=q, gold=gold, condition=cond.to_dict()))

        phrases = [c.phrase() for c in chosen]
        if len(phrases) == 1:
            joined = phrases[0]
        else:
            joined = ", ".join(phrases[:-1]) + ", and " + phrases[-1]
        question = f"Which contracts in the collection {joined}?"

        doc_ids = sorted(by_id)
        total_chars = sum(len(by_id[d].text) for d in doc_ids)
        return Task(
            task_id=task_id,
            domain="legal",
            instructions=build_instructions(n_docs, chosen, descriptions),
            question=question,
            doc_ids=doc_ids,
            steps=steps,
            metadata={
                "n_docs": n_docs,
                "n_steps": n_steps,
                "total_chars": total_chars,
                "approx_tokens": total_chars // 4,
                "n_gold": len(golds[-1]),
                "source": "CUAD v1",
            },
        )
    return None


def generate(
    contracts: list[Contract],
    descriptions: dict[str, str],
    n_docs_list: list[int],
    n_steps_list: list[int],
    n_tasks: int,
    seed: int,
) -> list[Task]:
    # Drop contracts whose governing law clause is redacted or unclear, so
    # governing-law gold answers are never ambiguous.
    pool = [c for c in contracts if not c.governing_law_unparsed]
    tasks = []
    for n_docs in n_docs_list:
        for n_steps in n_steps_list:
            rng = random.Random(f"{seed}-{n_docs}-{n_steps}")
            made = 0
            for k in range(n_tasks):
                tid = f"legal-d{n_docs}-s{n_steps}-{k:03d}"
                t = make_task(pool, n_docs, n_steps, rng, tid, descriptions)
                if t is None:
                    print(f"warning: could not build {tid}")
                    continue
                tasks.append(t)
                made += 1
            print(f"n_docs={n_docs:4d} n_steps={n_steps}: {made} tasks")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuad-dir", default="data/raw/cuad", help="folder with CUADv1.json and category_descriptions.csv")
    ap.add_argument("--out", default="data/legal")
    ap.add_argument("--n-docs", type=int, nargs="+", default=[10, 50, 200])
    ap.add_argument("--n-steps", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--n-tasks", type=int, default=10, help="tasks per (n_docs, n_steps) cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cuad_dir = Path(args.cuad_dir)
    contracts = load_cuad(cuad_dir / "CUADv1.json")
    descriptions = load_category_descriptions(cuad_dir / "category_descriptions.csv")

    out = Path(args.out)
    store = DocStore(out)
    for c in contracts:
        store.write(c.doc_id, c.text)
    (out / "contracts.json").write_text(
        json.dumps(
            [{"doc_id": c.doc_id, "title": c.title, "governing_law": c.governing_law} for c in contracts],
            indent=1,
        )
    )

    tasks = generate(contracts, descriptions, args.n_docs, args.n_steps, args.n_tasks, args.seed)
    save_tasks(tasks, out / "tasks.jsonl")

    kinds = Counter(s.condition["kind"] for t in tasks for s in t.steps)
    print(f"wrote {len(tasks)} tasks to {out / 'tasks.jsonl'} ({dict(kinds)} conditions)")


if __name__ == "__main__":
    main()
