"""Document storage and per-task workspaces.

All documents for a testbed live once in `<root>/docs/<doc_id>.txt`. For a run,
`materialize` creates a workspace that holds only the task's collection, so an
agent (or the harness's code tool) sees exactly the documents it should.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from core.task import Task


class DocStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.docs_dir = self.root / "docs"

    def path(self, doc_id: str) -> Path:
        return self.docs_dir / f"{doc_id}.txt"

    def read(self, doc_id: str) -> str:
        return self.path(doc_id).read_text()

    def write(self, doc_id: str, text: str) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.path(doc_id).write_text(text)

    def ids(self) -> list[str]:
        return sorted(p.stem for p in self.docs_dir.glob("*.txt"))

    def size_chars(self, doc_ids: list[str]) -> int:
        return sum(self.path(d).stat().st_size for d in doc_ids)


def materialize(task: Task, store: DocStore, workdir: str | Path, copy: bool = False) -> Path:
    """Build a workspace for one task.

    Layout:
        workdir/task.json     instructions and question (no gold answers)
        workdir/docs/*.txt    the task's documents only
    """
    workdir = Path(workdir)
    docs = workdir / "docs"
    if docs.exists():
        shutil.rmtree(docs)
    docs.mkdir(parents=True)
    for doc_id in task.doc_ids:
        src = store.path(doc_id).resolve()
        dst = docs / f"{doc_id}.txt"
        if copy:
            shutil.copyfile(src, dst)
        else:
            os.symlink(src, dst)
    public = {
        "task_id": task.task_id,
        "instructions": task.instructions,
        "question": task.question,
        "steps": [s.question for s in task.steps],
        "doc_ids": task.doc_ids,
    }
    (workdir / "task.json").write_text(json.dumps(public, indent=2))
    return workdir
