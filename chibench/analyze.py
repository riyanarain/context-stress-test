"""Summarize an audit CSV: per-submission tables and task-stratified comparisons.

Comparisons are stratified by task: for each task we compare trials with and
without a context signal (across submissions), then average the differences
over tasks that have both kinds of trials. This removes differences in task
difficulty, but it is still correlational: stronger agents may both pass more
and trigger a signal less often.

Usage:
    python -m chibench.analyze data/chibench/audit_public.csv --out reports/chibench_public.md
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path: str | Path) -> list[dict]:
    rows = [r for r in csv.DictReader(open(path)) if not r.get("error")]
    for r in rows:
        for k in ("reward", "fractional_reward", "n_tool_calls", "n_app_calls", "handbook_reads", "handbook_docs",
                  "handbook_rereads", "obs_chars_total", "obs_chars_max", "truncation_events", "tokens_hidden",
                  "offload_events", "offload_never_read", "peak_prompt_tokens", "n_input_tokens", "cost_usd", "n_steps"):
            r[k] = _f(r.get(k))
    return [r for r in rows if r["reward"] is not None]


def drop_duplicate_submissions(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Some packets are re-uploads of the same trials. Keep the latest copy."""
    sig = defaultdict(set)
    for r in rows:
        sig[r["submission"]].add((r["task"], r["n_input_tokens"], r["reward"]))
    seen, dropped = {}, []
    for sub in sorted(sig, reverse=True):
        key = frozenset(sig[sub])
        if key in seen:
            dropped.append(sub)
        else:
            seen[key] = sub
    return [r for r in rows if r["submission"] not in dropped], dropped


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def per_submission(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["submission"]].append(r)
    out = []
    for sub, rs in sorted(g.items(), key=lambda kv: -statistics.mean(r["reward"] for r in kv[1])):
        costs = [r["cost_usd"] for r in rs if r["cost_usd"]]
        out.append({
            "submission": sub, "agent": rs[0]["agent"], "model": rs[0]["model"], "n": len(rs),
            "pass": statistics.mean(r["reward"] for r in rs),
            "frac": statistics.mean(r["fractional_reward"] or 0 for r in rs),
            "tool_calls": med(r["n_tool_calls"] for r in rs),
            "input_tok": med(r["n_input_tokens"] for r in rs),
            "peak_prompt": med(r["peak_prompt_tokens"] for r in rs),
            "trunc": statistics.mean(1.0 if (r["truncation_events"] or 0) > 0 else 0.0 for r in rs),
            "offload": statistics.mean(1.0 if (r["offload_events"] or 0) > 0 else 0.0 for r in rs),
            "hb_docs": med(r["handbook_docs"] for r in rs),
            "cost": statistics.mean(costs) if costs else None,
        })
    return out


def stratified_diff(rows, signal, n_boot=1000, seed=0, strata=lambda r: r["task"]):
    """Mean over strata (default: tasks) of pass(signal) - pass(no signal), with a bootstrap CI over strata."""
    by_task = defaultdict(lambda: ([], []))
    for r in rows:
        s = signal(r)
        if s is None:
            continue
        by_task[strata(r)][0 if s else 1].append(r["reward"])
    diffs = [statistics.mean(a) - statistics.mean(b) for a, b in by_task.values() if a and b]
    n_with = sum(len(a) for a, b in by_task.values() if a and b)
    n_without = sum(len(b) for a, b in by_task.values() if a and b)
    if not diffs:
        return None
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {"diff": statistics.mean(diffs), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot) - 1],
            "tasks": len(diffs), "n_with": n_with, "n_without": n_without}


def task_median(rows, key, strata=lambda r: r["task"]):
    m = defaultdict(list)
    for r in rows:
        if r[key] is not None:
            m[strata(r)].append(r[key])
    return {t: statistics.median(v) for t, v in m.items()}


def build_report(rows, dropped) -> str:
    L = []
    L.append("# CHI-Bench public trajectories: context audit\n")
    L.append(f"{len(rows)} trials from {len({r['submission'] for r in rows})} leaderboard submissions "
             f"({', '.join(sorted({r['domain'] for r in rows}))}). Duplicate packets dropped: "
             f"{', '.join(dropped) or 'none'}.\n")

    L.append("## Per submission\n")
    L.append("| submission | agent | pass@1 | mean partial credit | median tool calls | median input tokens | median peak prompt | trials with truncation | trials with offload | median handbook docs | mean cost |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in per_submission(rows):
        L.append(f"| {s['submission']} | {s['agent']} | {s['pass']:.2f} | {s['frac']:.2f} | {fmt(s['tool_calls'],0)} | "
                 f"{fmt(s['input_tok'],0)} | {fmt(s['peak_prompt'],0)} | {s['trunc']:.0%} | {s['offload']:.0%} | "
                 f"{fmt(s['hb_docs'],0)} | {('$' + fmt(s['cost'])) if s['cost'] else 'n/a'} |")

    L.append("\n## Context signals vs passing, within the same task\n")
    L.append("Positive means trials with the signal pass more often than trials of the same task without it. "
             "95% intervals come from resampling tasks.\n")
    med_in = task_median(rows, "n_input_tokens")
    med_peak = task_median(rows, "peak_prompt_tokens")
    med_docs = task_median(rows, "handbook_docs")
    med_calls = task_median(rows, "n_tool_calls")
    signals = [
        ("any tool output truncated", lambda r: (r["truncation_events"] or 0) > 0),
        ("any output offloaded to a file", lambda r: (r["offload_events"] or 0) > 0),
        ("offloaded output never opened", lambda r: (r["offload_never_read"] or 0) > 0 if (r["offload_events"] or 0) > 0 else None),
        ("input tokens above task median", lambda r: None if r["n_input_tokens"] is None else r["n_input_tokens"] > med_in.get(r["task"], 0)),
        ("peak prompt above task median", lambda r: None if r["peak_prompt_tokens"] is None else r["peak_prompt_tokens"] > med_peak.get(r["task"], 0)),
        ("tool calls above task median", lambda r: None if r["n_tool_calls"] is None else r["n_tool_calls"] > med_calls.get(r["task"], 0)),
        ("handbook docs above task median", lambda r: None if r["handbook_docs"] is None else r["handbook_docs"] > med_docs.get(r["task"], 0)),
        ("re-read a handbook doc", lambda r: (r["handbook_rereads"] or 0) > 0),
    ]
    L.append("| signal | pass-rate difference | 95% interval | tasks | trials with | trials without |")
    L.append("|---|---|---|---|---|---|")
    for name, fn in signals:
        d = stratified_diff(rows, fn)
        if d is None:
            L.append(f"| {name} | n/a | | | | |")
            continue
        L.append(f"| {name} | {d['diff']:+.3f} | [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['tasks']} | {d['n_with']} | {d['n_without']} |")

    L.append("\nMost of these compare different agents on the same task, so stronger agents can drive the result. "
             "Truncation is set by the harness (Codex truncates long outputs in nearly every trial), so the next table "
             "compares trials within the same task and the same harness.\n")
    th = lambda r: (r["task"], r["agent"])
    med_hidden = task_median([r for r in rows if (r["tokens_hidden"] or 0) > 0], "tokens_hidden", th)
    med_peak_h = task_median(rows, "peak_prompt_tokens", th)
    med_in_h = task_median(rows, "n_input_tokens", th)
    within = [
        ("tokens hidden by truncation above median (Codex only)",
         lambda r: None if r["agent"] != "codex" or not r["tokens_hidden"] else r["tokens_hidden"] > med_hidden.get(th(r), 0)),
        ("any tool output truncated (openai-agents and hermes)",
         lambda r: None if r["agent"] not in ("openai-agents", "hermes") else (r["truncation_events"] or 0) > 0),
        ("any output offloaded to a file (Claude Code)",
         lambda r: None if r["agent"] != "claude-code" else (r["offload_events"] or 0) > 0),
        ("peak prompt above median",
         lambda r: None if r["peak_prompt_tokens"] is None else r["peak_prompt_tokens"] > med_peak_h.get(th(r), 0)),
        ("input tokens above median",
         lambda r: None if not r["n_input_tokens"] else r["n_input_tokens"] > med_in_h.get(th(r), 0)),
    ]
    L.append("| signal (within task and harness) | pass-rate difference | 95% interval | strata | trials with | trials without |")
    L.append("|---|---|---|---|---|---|")
    for name, fn in within:
        d = stratified_diff(rows, fn, strata=th)
        if d is None:
            L.append(f"| {name} | n/a | | | | |")
            continue
        L.append(f"| {name} | {d['diff']:+.3f} | [{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['tasks']} | {d['n_with']} | {d['n_without']} |")

    codex = [r for r in rows if r["agent"] == "codex" and r["tokens_hidden"]]
    if codex:
        L.append(f"\nIn Codex trials with truncation, the median trial hid {fmt(med(r['tokens_hidden'] for r in codex), 0)} "
                 f"tokens of tool output across {fmt(med(r['truncation_events'] for r in codex), 0)} truncated outputs.\n")

    L.append("\n## How failed trials fail\n")
    failed = [r for r in rows if r["reward"] == 0.0]
    fr = [r["fractional_reward"] or 0 for r in failed]
    n = max(1, len(fr))
    b1 = sum(1 for x in fr if x >= 0.9) / n
    b2 = sum(1 for x in fr if 0.7 <= x < 0.9) / n
    b3 = sum(1 for x in fr if x < 0.7) / n
    L.append(f"{len(failed)} failed trials. Share of checks passed in failed trials: at least 90% in {b1:.0%}, "
             f"70% to 90% in {b2:.0%}, under 70% in {b3:.0%}. A pass requires every check to pass.\n")
    L.append("| domain | failed trials | most common failed check groups (share of failed trials) |")
    L.append("|---|---|---|")
    for dom in sorted({r["domain"] for r in rows}):
        fs = [r for r in failed if r["domain"] == dom]
        c = Counter()
        for r in fs:
            for part in filter(None, (r["failed_stages"] or "").split(";")):
                c[part.rsplit(":", 1)[0]] += 1
        top = ", ".join(f"{k} ({v / max(1, len(fs)):.0%})" for k, v in c.most_common(5))
        L.append(f"| {dom} | {len(fs)} | {top} |")
    L.append("\nCaveat: these are correlations across existing runs. They point to where to look, not to causes. "
             "The controlled runs (same agent, context policy changed) are what test causes.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows, dropped = drop_duplicate_submissions(load(args.audit_csv))
    report = build_report(rows, dropped)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
