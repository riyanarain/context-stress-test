from core import Step, Task, extract_ids, load_tasks, save_tasks, score_set, score_task
from core.docstore import DocStore, materialize


def make_task():
    return Task(
        task_id="t1",
        domain="test",
        instructions="inst",
        question="q",
        doc_ids=["C0001", "C0002", "C0003"],
        steps=[Step("s1", ["C0001", "C0002"]), Step("s2", ["C0002"])],
    )


def test_extract_ids_uses_last_answer_marker():
    text = "I looked at C0001 and C0003.\nANSWER: C0002, c0001, C9999"
    assert extract_ids(text, ["C0001", "C0002", "C0003"]) == {"C0001", "C0002"}


def test_extract_ids_without_marker_and_none():
    assert extract_ids("C0003 only", ["C0003"]) == {"C0003"}
    assert extract_ids("ANSWER: none", ["C0003"]) == set()
    assert extract_ids(None, ["C0003"]) == set()


def test_score_set():
    s = score_set({"a", "b"}, {"b", "c"})
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["exact"] == 0.0
    assert score_set(set(), set())["exact"] == 1.0
    assert score_set(set(), {"a"})["f1"] == 0.0


def test_score_task_single_and_multi():
    t = make_task()
    assert score_task(t, "ANSWER: C0002")["final"]["exact"] == 1.0
    multi = score_task(t, ["ANSWER: C0001, C0002"])
    assert multi["steps"][0]["exact"] == 1.0
    assert multi["final"]["recall"] == 0.0  # missing step counts as empty


def test_roundtrip_and_materialize(tmp_path):
    t = make_task()
    save_tasks([t], tmp_path / "tasks.jsonl")
    (loaded,) = load_tasks(tmp_path / "tasks.jsonl")
    assert loaded == t and loaded.gold == ["C0002"]

    store = DocStore(tmp_path / "store")
    for d in t.doc_ids + ["C0004"]:
        store.write(d, f"text of {d}")
    work = materialize(t, store, tmp_path / "work")
    assert sorted(p.stem for p in (work / "docs").iterdir()) == t.doc_ids
    assert "C0002" in (work / "docs" / "C0002.txt").read_text()
    assert "gold" not in (work / "task.json").read_text()
