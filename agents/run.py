"""Run agents over EHR tasks with one or more context policies.

Examples:
    # Pipeline check, no API key needed
    python -m agents.run --client oracle --out runs/oracle-check

    # Real run (reads LLM_API_KEY, LLM_BASE_URL, LLM_MODEL from the environment or .env)
    python -m agents.run --families update --levels 0 2 --limit 2 --policies full truncate summary \
        --out runs/pilot

Results go to <out>/results.jsonl (one line per episode) and full traces to
<out>/traces/<task_id>__<policy>.jsonl. Rerunning the same command skips
episodes that already have a result, so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents.llm import OpenAIChatClient, OracleClient
from agents.loop import run_episode
from agents.policies import POLICIES, make_policy
from ehr.families import FAMILIES


def load_tasks(task_dir: Path, families, levels, limit) -> list[dict]:
    tasks = []
    for f in sorted(task_dir.glob("*.json")):
        t = json.loads(f.read_text())
        if t["family"] in families and t["level"] in levels:
            tasks.append(t)
    if limit:
        seen: dict = {}
        kept = []
        for t in tasks:
            k = (t["family"], t["level"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= limit:
                kept.append(t)
        tasks = kept
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="data/ehr/tasks")
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0, help="max tasks per (family, level); 0 = all")
    ap.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    ap.add_argument("--client", choices=["openai", "oracle"], default="openai")
    ap.add_argument("--model", default=None, help="overrides LLM_MODEL")
    ap.add_argument("--base-url", default=None, help="overrides LLM_BASE_URL")
    ap.add_argument("--temperature", type=float, default=0.0, help="use -1 to omit (some reasoning models)")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--truncate-chars", type=int, default=3000)
    ap.add_argument("--summary-budget", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens, for cost estimates")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks), args.families, args.levels, args.limit)
    if not tasks:
        raise SystemExit(f"no tasks found in {args.tasks}; run python -m ehr.generate first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task_id"], r["policy"]))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.client == "openai":
        temp = None if args.temperature is not None and args.temperature < 0 else args.temperature
        client = OpenAIChatClient(model=args.model, base_url=args.base_url, temperature=temp)
        factory = None
    else:
        client, factory = None, OracleClient

    jobs = [(t, p) for p in args.policies for t in tasks if (t["task_id"], p) not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already done)")
    lock = threading.Lock()

    def work(task, pname):
        policy = make_policy(pname, truncate_chars=args.truncate_chars, summary_budget=args.summary_budget)
        res = run_episode(task, client, policy, max_steps=args.max_steps,
                          trace_path=out / "traces" / f"{task['task_id']}__{pname}.jsonl", client_factory=factory)
        res["cost_usd"] = round(((res["prompt_tokens"] + res["summary_prompt_tokens"]) * args.price_in
                                 + (res["completion_tokens"] + res["summary_completion_tokens"]) * args.price_out) / 1e6, 4)
        with lock:
            with results_path.open("a") as f:
                f.write(json.dumps(res) + "\n")
        return res

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(work, t, p) for t, p in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            flag = f"  ERROR {r['error']}" if r["error"] else ""
            print(f"{r['task_id']:20s} {r['policy']:9s} f1={r['f1']:.2f} exact={r['exact']:.0f} "
                  f"steps={r['steps']:3d} prompt_tok={r['prompt_tokens']:,}{flag}")
    print(f"results in {results_path}; summarize with: python -m agents.report {out}")


if __name__ == "__main__":
    main()
