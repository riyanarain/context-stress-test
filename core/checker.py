"""Automatic answer checking for set-valued answers.

Agents answer in free text. We pull out every valid document ID that appears
in the answer and compare that set to the gold set.
"""

from __future__ import annotations

import re
from typing import Iterable

from core.task import Task

ANSWER_MARKER = re.compile(r"ANSWER\s*:", re.IGNORECASE)


def extract_ids(text: str, valid_ids: Iterable[str]) -> set[str]:
    """Return the valid IDs mentioned in `text`.

    If the text contains an 'ANSWER:' marker, only the part after the last
    marker is used, so IDs the agent mentions while reasoning are not counted.
    """
    if text is None:
        return set()
    parts = ANSWER_MARKER.split(text)
    if len(parts) > 1:
        text = parts[-1]
    valid = set(valid_ids)
    found = re.findall(r"[A-Za-z]+\d+", text)
    return {tok.upper() for tok in found if tok.upper() in valid}


def score_set(pred: Iterable[str], gold: Iterable[str]) -> dict:
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    if not pred and not gold:
        precision = recall = 1.0
    else:
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact": float(pred == gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(pred),
        "n_gold": len(gold),
    }


def score_task(task: Task, answers: str | list[str]) -> dict:
    """Score a single-turn answer (str) or a multi-turn answer (one str per step).

    The final score always uses the last answer against the task gold. For
    multi-turn runs, per-step scores show where in the horizon errors start.
    """
    if isinstance(answers, str):
        final = score_set(extract_ids(answers, task.doc_ids), task.gold)
        return {"mode": "single", "final": final}

    per_step = []
    for step, ans in zip(task.steps, answers):
        per_step.append(score_set(extract_ids(ans, task.doc_ids), step.gold))
    # Missing answers for later steps count as empty.
    for step in task.steps[len(answers):]:
        per_step.append(score_set(set(), step.gold))
    return {"mode": "multi", "final": per_step[-1], "steps": per_step}
