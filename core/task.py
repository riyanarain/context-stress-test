"""Task format shared across domains.

A task gives the agent a collection of documents (by ID) and asks a question
whose answer is a set of document IDs. Every task can be run two ways:

- single-turn: the agent gets `question`, which states all conditions at once.
- multi-turn: the agent gets `steps` one at a time. Each step narrows the
  previous answer, so later steps depend on details found in earlier ones.
  This is the horizon knob.

The gold answer for the whole task is the gold of the last step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Step:
    question: str
    gold: list[str]
    condition: dict = field(default_factory=dict)  # structured form, used by the audit


@dataclass
class Task:
    task_id: str
    domain: str
    instructions: str
    question: str
    doc_ids: list[str]
    steps: list[Step]
    metadata: dict = field(default_factory=dict)

    @property
    def gold(self) -> list[str]:
        return self.steps[-1].gold

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        steps = [Step(**s) for s in d["steps"]]
        return cls(**{**d, "steps": steps})


def save_tasks(tasks: Iterable[Task], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_tasks(path: str | Path) -> list[Task]:
    with Path(path).open() as f:
        return [Task.from_dict(json.loads(line)) for line in f if line.strip()]
