"""Context management policies.

Each policy can change a tool output before it enters the history
(process_tool_output) and can rewrite the history before each model call
(before_call). Every change is reported as an event for the trace.

full       keep everything (upper bound on context use)
truncate   cap each tool output, keeping the head and tail
summary    when the history passes a token budget, replace older turns with a
           model-written summary, keeping the most recent turns verbatim
"""

from __future__ import annotations

import json


def est_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


class FullContext:
    name = "full"

    def config(self) -> dict:
        return {"name": self.name}

    def process_tool_output(self, text: str) -> tuple[str, dict | None]:
        return text, None

    def before_call(self, messages: list[dict], client) -> list[dict]:
        return []


class Truncate(FullContext):
    name = "truncate"

    def __init__(self, max_chars: int = 3000, head_frac: float = 0.67):
        self.max_chars = max_chars
        self.head_frac = head_frac

    def config(self) -> dict:
        return {"name": self.name, "max_chars": self.max_chars, "head_frac": self.head_frac}

    def process_tool_output(self, text: str):
        if len(text) <= self.max_chars:
            return text, None
        head = int(self.max_chars * self.head_frac)
        tail = self.max_chars - head
        cut = len(text) - head - tail
        out = text[:head] + f"\n[... {cut} characters omitted ...]\n" + text[-tail:]
        return out, {"event": "truncate", "original_chars": len(text), "kept_chars": head + tail}


SUMMARY_PROMPT = """You are helping an agent that works in an EHR system keep track of its progress.
Below is the agent's task and the older part of its work log. Write a concise summary that the agent
will use instead of the log. Keep every detail that may matter later: patient IDs, lab values with dates,
medication and allergy facts, provider IDs, action IDs, what has been done, and what is still left.

TASK:
{task}

PREVIOUS SUMMARY (may be empty):
{previous}

OLDER WORK LOG:
{log}

SUMMARY:"""


class RollingSummary(FullContext):
    name = "summary"

    def __init__(self, budget_tokens: int = 16000, keep_recent: int = 6, max_log_chars: int = 60000):
        self.budget_tokens = budget_tokens
        self.keep_recent = keep_recent
        self.max_log_chars = max_log_chars

    def config(self) -> dict:
        return {"name": self.name, "budget_tokens": self.budget_tokens, "keep_recent": self.keep_recent}

    @staticmethod
    def _render(msgs: list[dict], limit: int) -> str:
        parts = []
        for m in msgs:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    parts.append(f"CALL {tc['function']['name']}({tc['function']['arguments']})")
                if m.get("content"):
                    parts.append(f"AGENT: {m['content']}")
            elif m["role"] == "tool":
                parts.append(f"RESULT: {m['content']}")
            else:
                parts.append(f"{m['role'].upper()}: {m['content']}")
        text = "\n".join(parts)
        if len(text) > limit:  # the summarizer has a context limit too
            text = text[: limit // 2] + "\n[... log shortened ...]\n" + text[-limit // 2:]
        return text

    def before_call(self, messages: list[dict], client) -> list[dict]:
        if est_tokens(messages) <= self.budget_tokens:
            return []
        # messages: [system, task, (summary)?, assistant, tool, ..., assistant, tool, ...]
        start = 2
        previous = ""
        if len(messages) > 2 and messages[2]["role"] == "user" and messages[2]["content"].startswith("[Summary"):
            previous = messages[2]["content"]
            start = 3
        group_starts = [i for i in range(start, len(messages)) if messages[i]["role"] == "assistant"]
        if len(group_starts) <= self.keep_recent:
            return []
        cut = group_starts[-self.keep_recent]
        old = messages[start:cut]
        if not old:
            return []
        before = est_tokens(messages)
        prompt = SUMMARY_PROMPT.format(task=messages[1]["content"], previous=previous,
                                       log=self._render(old, self.max_log_chars))
        summary, usage = client.complete(prompt)
        new = {"role": "user", "content": "[Summary of earlier work]\n" + summary.strip()}
        messages[2:cut] = [new]
        return [{"event": "summarize", "removed_messages": len(old), "tokens_before": before,
                 "tokens_after": est_tokens(messages), "summary_usage": usage, "summary": summary}]


def make_policy(name: str, **kw):
    if name == "full":
        return FullContext()
    if name == "truncate":
        return Truncate(max_chars=kw.get("truncate_chars", 3000))
    if name == "summary":
        return RollingSummary(budget_tokens=kw.get("summary_budget", 16000), keep_recent=kw.get("keep_recent", 6))
    raise ValueError(f"unknown policy {name!r}")


POLICIES = ["full", "truncate", "summary"]
