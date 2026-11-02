"""Domain-independent generator for chained-condition tasks.

A testbed supplies a pool of documents and a list of conditions. This module
samples a collection, then picks a chain of conditions where every step
strictly shrinks the candidate set and leaves at least one document. Each
testbed only defines what its conditions mean and how they are phrased.

A condition needs:
    kind: str, key: str, positive: bool
    holds(doc) -> bool
    phrase() -> str          e.g. "contain a non-compete clause"
    to_dict() -> dict
A document needs `doc_id` and `text`.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

from core.task import Step, Task


def default_compatible(chosen: list, cond) -> bool:
    """Do not reuse the same (kind, key) twice in one chain."""
    return all((c.kind, c.key) != (cond.kind, cond.key) for c in chosen)


def make_chain_task(
    pool: Sequence[Any],
    conditions: Sequence[Any],
    n_docs: int,
    n_steps: int,
    rng: random.Random,
    *,
    task_id: str,
    domain: str,
    noun: str,
    build_instructions: Callable[[int, list], str],
    compatible: Callable[[list, Any], bool] = default_compatible,
    kind_boost: dict[str, float] | None = None,
    keep_range: tuple[float, float] = (0.3, 0.85),
    metadata: dict | None = None,
    max_attempts: int = 500,
) -> Task | None:
    """Build one task, or return None if no valid chain is found.

    kind_boost maps a condition kind to the probability of forcing that kind
    at a step (when it is available), so rare kinds still show up.
    keep_range is the preferred fraction of candidates each step keeps, which
    keeps chains from collapsing to one document too early.
    """
    kind_boost = kind_boost or {}
    for _ in range(max_attempts):
        docs = rng.sample(list(pool), n_docs)
        by_id = {d.doc_id: d for d in docs}
        current = set(by_id)
        chosen: list = []
        golds: list[list[str]] = []
        for s in range(n_steps):
            remaining = n_steps - s
            options = []
            for cond in conditions:
                if not compatible(chosen, cond):
                    continue
                new = {d for d in current if cond.holds(by_id[d])}
                if remaining <= len(new) < len(current):
                    options.append((cond, new))
            if not options:
                break
            lo, hi = keep_range
            good = [(c, n) for c, n in options if lo <= len(n) / len(current) <= hi] or options
            for kind, p in kind_boost.items():
                sub = [(c, n) for c, n in good if c.kind == kind]
                if sub and rng.random() < p:
                    good = sub
                    break
            cond, new = rng.choice(good)
            chosen.append(cond)
            current = new
            golds.append(sorted(new))
        if len(chosen) != n_steps:
            continue

        steps = []
        for i, (cond, gold) in enumerate(zip(chosen, golds)):
            if i == 0:
                q = f"Which {noun} in the collection {cond.phrase()}?"
            else:
                q = f"Of the {noun} in your previous answer, which {cond.phrase()}?"
            steps.append(Step(question=q, gold=gold, condition=cond.to_dict()))

        phrases = [c.phrase() for c in chosen]
        joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + ", and " + phrases[-1]

        doc_ids = sorted(by_id)
        total_chars = sum(len(by_id[d].text) for d in doc_ids)
        return Task(
            task_id=task_id,
            domain=domain,
            instructions=build_instructions(n_docs, chosen),
            question=f"Which {noun} in the collection {joined}?",
            doc_ids=doc_ids,
            steps=steps,
            metadata={
                "n_docs": n_docs,
                "n_steps": n_steps,
                "total_chars": total_chars,
                "approx_tokens": total_chars // 4,
                "n_gold": len(golds[-1]),
                **(metadata or {}),
            },
        )
    return None


def generate_grid(
    make: Callable[[int, int, random.Random, str], Task | None],
    prefix: str,
    n_docs_list: list[int],
    n_steps_list: list[int],
    n_tasks: int,
    seed: int,
) -> list[Task]:
    """Generate n_tasks per (n_docs, n_steps) cell with a separate seeded RNG per cell."""
    tasks = []
    for n_docs in n_docs_list:
        for n_steps in n_steps_list:
            rng = random.Random(f"{seed}-{n_docs}-{n_steps}")
            made = 0
            for k in range(n_tasks):
                tid = f"{prefix}-d{n_docs}-s{n_steps}-{k:03d}"
                t = make(n_docs, n_steps, rng, tid)
                if t is None:
                    print(f"warning: could not build {tid}")
                    continue
                tasks.append(t)
                made += 1
            print(f"n_docs={n_docs:4d} n_steps={n_steps}: {made} tasks")
    return tasks
