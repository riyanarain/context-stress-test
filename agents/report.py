"""Summarize a run directory: accuracy and cost by family, level, and policy.

Usage:
    python -m agents.report runs/pilot            # prints a table
    python -m agents.report runs/pilot --csv      # also writes runs/pilot/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["level"], r["policy"])].append(r)
    out = []
    for (fam, lvl, pol), rs in sorted(groups.items()):
        out.append({
            "family": fam,
            "level": lvl,
            "policy": pol,
            "n": len(rs),
            "exact": round(statistics.mean(r["exact"] for r in rs), 3),
            "f1": round(statistics.mean(r["f1"] for r in rs), 3),
            "finished": round(statistics.mean(1.0 if r["finished"] else 0.0 for r in rs), 3),
            "errors": sum(1 for r in rs if r["error"]),
            "steps": round(statistics.mean(r["steps"] for r in rs), 1),
            "prompt_tokens": int(statistics.mean(r["prompt_tokens"] + r.get("summary_prompt_tokens", 0) for r in rs)),
            "max_context": int(statistics.mean(r["max_context_tokens"] for r in rs)),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rs), 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()
    path = Path(args.run_dir) / "results.jsonl"
    latest = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["task_id"], r["policy"])] = r  # reruns replace earlier attempts
    rows = list(latest.values())
    table = summarize(rows)
    cols = ["family", "level", "policy", "n", "exact", "f1", "finished", "errors", "steps", "prompt_tokens",
            "max_context", "cost_usd"]
    print(" | ".join(cols))
    print(" | ".join("---" for _ in cols))
    for r in table:
        print(" | ".join(str(r[c]) for c in cols))
    total = sum(r.get("cost_usd", 0.0) for r in rows)
    print(f"\n{len(rows)} episodes, estimated cost ${total:.2f}")
    if args.csv:
        with (Path(args.run_dir) / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(table)


if __name__ == "__main__":
    main()
