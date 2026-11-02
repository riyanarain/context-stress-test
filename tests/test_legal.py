import random

from testbeds.legal.cuad import Contract, parse_governing_law
from testbeds.legal.generate import make_task


def test_parse_governing_law():
    assert parse_governing_law("governed by the laws of the State of Delaware") == "Delaware"
    assert parse_governing_law("THE LAWS OF THE COMMONWEALTH OF PENNSYLVANIA") == "Pennsylvania"
    assert parse_governing_law("the laws of West Virginia") == "West Virginia"
    # Venue named before the governing law should not win.
    assert parse_governing_law(
        "Courts in Texas have jurisdiction. This Agreement is governed by the laws of New York."
    ) == "New York"
    assert parse_governing_law("governed by English law") == "England and Wales"
    assert parse_governing_law("the laws of [***]") is None


def synthetic_pool(n=60, seed=0):
    rng = random.Random(seed)
    cats = ["Audit Rights", "Exclusivity", "Non-Compete", "Cap On Liability", "License Grant"]
    pool = []
    for i in range(n):
        clauses = {c: (["span"] if rng.random() < 0.5 else []) for c in cats}
        law = rng.choice(["New York", "Delaware", "California", None])
        pool.append(Contract(f"C{i:04d}", f"title {i}", "x" * 100, clauses, law))
    return pool


def test_make_task_steps_shrink_and_gold_is_correct():
    pool = synthetic_pool()
    by_id = {c.doc_id: c for c in pool}
    for n_docs, n_steps in [(10, 1), (10, 3), (30, 5)]:
        t = make_task(pool, n_docs, n_steps, random.Random(1), "t", {})
        assert t is not None and t.n_docs == n_docs and t.n_steps == n_steps
        prev = set(t.doc_ids)
        for s in t.steps:
            gold = set(s.gold)
            assert gold and gold < prev  # non-empty and strictly smaller
            c = s.condition
            for d in prev:
                con = by_id[d]
                val = bool(con.clauses.get(c["key"])) if c["kind"] == "clause" else con.governing_law == c["key"]
                assert (d in gold) == (val if c["positive"] else not val)
            prev = gold
        assert t.gold == t.steps[-1].gold


def test_make_task_is_deterministic():
    pool = synthetic_pool()
    a = make_task(pool, 10, 3, random.Random(7), "t", {})
    b = make_task(pool, 10, 3, random.Random(7), "t", {})
    assert a == b
