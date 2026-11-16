"""Extract context and outcome features from CHI-Bench trials, one row per trial.

Context signals:
  truncation_events     tool outputs the harness cut (Codex: "truncated output
                        (original token count: N)"); tokens_hidden sums N
  offload_events        large outputs the harness saved to a file instead of
                        showing them (Claude Code: "Output too large ... saved to")
  offload_never_read    offloaded outputs the agent never opened afterwards
  handbook_reads        reads of handbook documents; handbook_docs counts distinct
                        documents, handbook_rereads counts repeat reads
  peak_prompt_tokens    largest per-call prompt, when the harness logs metrics
Outcome:
  reward (binary pass), fractional_reward, failed_checks, failed_stages

Usage:
    python -m chibench.audit <root> --out audit.csv
<root> can be a leaderboard submissions folder or CHI-Bench's logs/experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from chibench.trajectory import Trial, find_trials, load_trial

HANDBOOK = re.compile(r"managed-care-operations-handbook/([\w./-]+?\.md)")
CODEX_TRUNC = re.compile(r"truncated output \(original token count: (\d+)\)")
GENERIC_TRUNC = re.compile(r"(?i)\[\.\.\. ?\d+ (characters|chars|lines|tokens) (omitted|truncated)|output truncated")
OFFLOAD = re.compile(r"Output too large \(([\d.]+)(KB|MB)\)\. Full output saved to: (\S+?)(?=\\n|\s|$)")
LOCAL_TOOLS = {"exec", "exec_command", "Bash", "bash", "Read", "read_file", "Glob", "Grep", "Write", "Edit",
               "terminal", "search_files", "local-fs-read", "local-bash", "ToolSearch", "TodoWrite", "tool_search",
               "tool_describe", "write_file", "patch", "shell"}
EMBEDDED_APP_CALL = re.compile(r"tools\.(\w+__\w+|mcp__[\w-]+__\w+)\(")


def domain_of(trial: Trial) -> str:
    parts = trial.path.parts
    for d in ("pa_provider", "pa_um", "cm", "prior_auth_provider", "prior_auth_um", "care_management", "marathon"):
        if d in parts:
            return {"prior_auth_provider": "pa_provider", "prior_auth_um": "pa_um", "care_management": "cm"}.get(d, d)
    name = trial.result.get("task_name", "")
    if "_payer" in name or "mdreview" in name:
        return "pa_um"
    return "unknown"


def submission_of(trial: Trial, root: Path) -> str:
    rel = trial.path.relative_to(root).parts
    return rel[0] if len(rel) > 1 else root.name


def stage_of(check: str) -> str:
    if check.startswith("judge."):
        return "judge:" + check.split(".", 1)[1].split(":", 1)[0]
    return check.split(".", 1)[0]


def features(trial: Trial, root: Path) -> dict:
    calls = trial.calls
    ar = trial.result.get("agent_result") or {}
    agent_info = trial.result.get("agent_info") or {}
    model = (agent_info.get("model_info") or {}).get("name") or trial.header.get("agent", {}).get("model_name", "")

    hb = []
    n_app = 0
    obs_chars = []
    trunc_events = 0
    tokens_hidden = 0
    offloads: list[tuple[int, str]] = []
    for i, c in enumerate(calls):
        hb += HANDBOOK.findall(c.args_text)
        if c.name not in LOCAL_TOOLS:
            n_app += 1
        n_app += len(EMBEDDED_APP_CALL.findall(c.args_text))
        obs_chars.append(len(c.output))
        hidden = [int(x) for x in CODEX_TRUNC.findall(c.output)]
        trunc_events += len(hidden) + len(GENERIC_TRUNC.findall(c.output))
        tokens_hidden += sum(hidden)
        for m in OFFLOAD.finditer(c.output):
            offloads.append((i, m.group(3)))
    later_args = [c.args_text for c in calls]
    never_read = 0
    for i, p in offloads:
        fname = p.rsplit("/", 1)[-1]
        if not any(fname in a for a in later_args[i + 1:]):
            never_read += 1

    prompts = [s["metrics"].get("prompt_tokens") or 0 for s in trial.steps if isinstance(s.get("metrics"), dict)]
    hb_counts = Counter(hb)
    sc = trial.scorecard
    failed = [k for k, v in (sc.get("check_scores") or {}).items() if v == 0.0]
    stages = Counter(stage_of(k) for k in failed)
    ts = [s.get("timestamp") for s in trial.steps if s.get("timestamp")]
    return {
        "submission": submission_of(trial, root),
        "agent": trial.header.get("agent", {}).get("name") or (trial.result.get("config") or {}).get("agent", {}).get("name", ""),
        "model": model,
        "domain": domain_of(trial),
        "task": trial.result.get("task_name", "").replace("actava-ai/", ""),
        "trial": trial.path.name,
        "reward": trial.reward,
        "fractional_reward": sc.get("fractional_reward"),
        "n_checks": sc.get("total_checks"),
        "failed_checks": ";".join(failed),
        "failed_stages": ";".join(f"{k}:{v}" for k, v in sorted(stages.items())),
        "n_steps": sum(1 for s in trial.steps if s.get("source") == "agent"),
        "n_tool_calls": len(calls),
        "n_app_calls": n_app,
        "handbook_reads": len(hb),
        "handbook_docs": len(hb_counts),
        "handbook_rereads": sum(v - 1 for v in hb_counts.values()),
        "obs_chars_total": sum(obs_chars),
        "obs_chars_max": max(obs_chars, default=0),
        "truncation_events": trunc_events,
        "tokens_hidden": tokens_hidden,
        "offload_events": len(offloads),
        "offload_never_read": never_read,
        "peak_prompt_tokens": max(prompts, default=None),
        "n_input_tokens": ar.get("n_input_tokens"),
        "n_cache_tokens": ar.get("n_cache_tokens"),
        "n_output_tokens": ar.get("n_output_tokens"),
        "cost_usd": ar.get("cost_usd"),
        "start": min(ts, default=""),
        "end": max(ts, default=""),
    }


def audit(root: str | Path, limit: int = 0) -> list[dict]:
    root = Path(root)
    rows = []
    for i, d in enumerate(find_trials(root)):
        if limit and i >= limit:
            break
        try:
            rows.append(features(load_trial(d), root))
        except Exception as e:  # keep going; report the bad trial
            rows.append({"submission": str(d.relative_to(root)), "trial": d.name, "error": f"{type(e).__name__}: {e}"})
    return rows


def write_csv(rows: list[dict], out: str | Path) -> None:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default="audit.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = audit(args.root, args.limit)
    write_csv(rows, args.out)
    errs = sum(1 for r in rows if "error" in r)
    print(f"audited {len(rows)} trials ({errs} errors) -> {args.out}")


if __name__ == "__main__":
    main()
