"""Estimate CHI-Bench run costs from public leaderboard costs.

Agent cost per trial comes from leaderboard trials whose model name matches
--model. The grader adds about $0.05 to $0.30 per trial (CHI-Bench docs/judge.md),
billed to the Anthropic key. Marathon cost is a rough guess: one session covers
25 tasks, so it is scaled from the single-task cost, which may be off in either direction.

Usage:
    python -m chibench.estimate data/chibench/audit_public.csv --model gpt-5.5
"""

from __future__ import annotations

import argparse
import statistics

from chibench.analyze import load

JUDGE_LOW, JUDGE_HIGH = 0.05, 0.30


def per_trial_cost(rows, model_substr: str) -> tuple[float, float, int] | None:
    costs = [r["cost_usd"] for r in rows if model_substr in (r["model"] or "") and r["cost_usd"]]
    if not costs:
        return None
    return statistics.median(costs), statistics.mean(costs), len(costs)


def plan(agent_cost: float) -> list[tuple[str, int, float, float]]:
    """(run, trials, low, high) for our planned runs."""
    out = []
    for name, n in [("smoke: 1 task per domain", 3), ("one domain, 25 tasks", 25), ("all 75 tasks", 75)]:
        out.append((name, n, n * (agent_cost + JUDGE_LOW), n * (agent_cost + JUDGE_HIGH)))
    out.append(("one Marathon session (rough)", 1, 25 * (agent_cost + JUDGE_LOW) * 0.5, 25 * (agent_cost + JUDGE_HIGH) * 1.5))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_csv")
    ap.add_argument("--model", default="gpt-5.5", help="substring of the model name, e.g. gpt-5.5 or luna")
    args = ap.parse_args()
    rows = load(args.audit_csv)
    c = per_trial_cost(rows, args.model)
    if c is None:
        models = sorted({r["model"] for r in rows if r["cost_usd"]})
        raise SystemExit(f"no cost data for {args.model!r}; models with costs: {models}")
    median, mean, n = c
    print(f"{args.model}: agent cost per trial median ${median:.2f}, mean ${mean:.2f} (from {n} public trials)")
    print(f"grader: ${JUDGE_LOW:.2f} to ${JUDGE_HIGH:.2f} per trial on the Anthropic key\n")
    print("run | trials | estimated total")
    for name, n_trials, lo, hi in plan(mean):
        print(f"{name} | {n_trials} | ${lo:,.0f} to ${hi:,.0f}")
    print("\nDouble these for two conditions (production Codex and Codex with the lab's context policy).")


if __name__ == "__main__":
    main()
