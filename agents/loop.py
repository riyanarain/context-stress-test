"""Run one agent episode on one task and log a full trace."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from agents.policies import est_tokens
from ehr.env import EHREnv, tool_schemas
from ehr.grade import grade

SYSTEM_PROMPT = """You are an operations agent working in a hospital's electronic health record (EHR) system.
All patient data is synthetic. Today's date is {today}.
Use the tools to complete the task. Actions only happen through tool calls; describing an action in text does not perform it.
When the task is complete, call finish."""

NUDGE = "Continue working with the tools. Call finish when the task is complete."


def run_episode(task: dict, client, policy, max_steps: int = 80, trace_path: str | Path | None = None,
                client_factory=None) -> dict:
    """Run one episode. If client_factory is given it is called with the env (used by OracleClient)."""
    env = EHREnv(task)
    if client_factory is not None:
        client = client_factory(env)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=task["today"])},
        {"role": "user", "content": task["instructions"]},
    ]
    tools = tool_schemas()
    trace = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0, "tool_calls": 0,
              "summary_prompt_tokens": 0, "summary_completion_tokens": 0, "max_context_tokens": 0}
    error = None
    nudges = 0
    t0 = time.time()
    step = 0
    try:
        for step in range(1, max_steps + 1):
            for ev in policy.before_call(messages, client):
                trace.append({"type": "policy", "step": step, **ev})
                u = ev.get("summary_usage") or {}
                totals["summary_prompt_tokens"] += u.get("prompt_tokens", 0)
                totals["summary_completion_tokens"] += u.get("completion_tokens", 0)
            ctx = est_tokens(messages)
            totals["max_context_tokens"] = max(totals["max_context_tokens"], ctx)
            msg, usage = client.chat(messages, tools)
            totals["model_calls"] += 1
            totals["prompt_tokens"] += usage["prompt_tokens"]
            totals["completion_tokens"] += usage["completion_tokens"]
            messages.append(msg)
            trace.append({"type": "model", "step": step, "context_tokens_est": ctx, "usage": usage,
                          "content": msg.get("content", ""), "tool_calls": msg.get("tool_calls", [])})
            calls = msg.get("tool_calls") or []
            if not calls:
                nudges += 1
                if nudges > 2:
                    break
                messages.append({"role": "user", "content": NUDGE})
                continue
            for tc in calls:
                name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                out = env.call(name, raw_args)
                totals["tool_calls"] += 1
                shown, ev = policy.process_tool_output(out)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": shown})
                rec = {"type": "tool", "step": step, "name": name, "args": raw_args, "output": out,
                       "output_chars": len(out), "shown_chars": len(shown)}
                if ev:
                    rec["policy_event"] = ev
                trace.append(rec)
                if env.done:
                    break
            if env.done:
                break
    except Exception as e:
        body = getattr(e, "body", None)
        error = f"{type(e).__name__}: {e}" + (f" | body: {json.dumps(body)[:800]}" if body else "")
        trace.append({"type": "error", "step": step, "error": error, "traceback": traceback.format_exc()})

    result = {
        "task_id": task["task_id"],
        "family": task["family"],
        "level": task["level"],
        "policy": policy.name,
        "policy_config": policy.config(),
        "model": getattr(client, "model", "unknown"),
        "steps": step,
        "error": error,
        "seconds": round(time.time() - t0, 1),
        **totals,
        **grade(env),
    }
    trace.append({"type": "result", **result})
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as f:
            for rec in trace:
                f.write(json.dumps(rec) + "\n")
    return result
