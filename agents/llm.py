"""Model clients.

OpenAIChatClient talks to any OpenAI-compatible chat completions endpoint
with tool calling: OpenAI, OpenRouter, Groq, Gemini's OpenAI-compatible
endpoint, Anthropic's OpenAI SDK compatibility endpoint, or a local server.
Configure it with environment variables (or a .env file in the repo root):

    LLM_API_KEY    API key
    LLM_BASE_URL   endpoint, e.g. https://openrouter.ai/api/v1 (omit for OpenAI)
    LLM_MODEL      model name

OracleClient needs no key. It solves tasks by reading hidden environment
state, and exists only to test the pipeline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class OpenAIChatClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float | None = 0.0, max_tokens: int = 8192, timeout: float = 300, max_retries: int = 4,
                 reasoning_effort: str | None = None):
        from openai import OpenAI

        load_dotenv()
        self.model = model or os.environ.get("LLM_MODEL")
        if not self.model:
            raise ValueError("set --model or LLM_MODEL")
        key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("set LLM_API_KEY (or OPENAI_API_KEY)")
        self.client = OpenAI(api_key=key, base_url=base_url or os.environ.get("LLM_BASE_URL") or None,
                             timeout=timeout, max_retries=0)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    def _create(self, **kw):
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                extra = {}
                if self.temperature is not None:
                    extra["temperature"] = self.temperature
                if self.reasoning_effort:
                    extra["reasoning_effort"] = self.reasoning_effort
                return self.client.chat.completions.create(model=self.model, max_tokens=self.max_tokens,
                                                           **extra, **kw)
            except Exception as e:  # retry rate limits and transient server errors
                status = getattr(e, "status_code", None)
                text = str(getattr(e, "body", "")) + str(e)
                daily_quota = status == 429 and ("per day" in text or "PerDay" in text)
                timed_out = type(e).__name__ == "APITimeoutError"
                if (attempt == self.max_retries or daily_quota or (timed_out and attempt >= 1)
                        or (status is not None and status < 500 and status != 429)):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def chat(self, messages: list[dict], tools: list[dict]) -> tuple[dict, dict]:
        resp = self._create(messages=messages, tools=tools, tool_choice="auto")
        m = resp.choices[0].message
        raw = m.model_dump(exclude_none=True)
        msg: dict = {"role": "assistant", "content": m.content or ""}
        # Keep provider-specific fields so they are sent back on the next request.
        # Gemini 3 models put a required "thought signature" in tool_calls[i].extra_content.
        for key in ("extra_content", "reasoning_content"):
            if key in raw:
                msg[key] = raw[key]
        if m.tool_calls:
            calls = []
            for tc in raw.get("tool_calls", []):
                tc = dict(tc)
                tc["type"] = "function"
                tc["function"] = {"name": tc["function"]["name"],
                                  "arguments": tc["function"].get("arguments") or "{}"}
                calls.append(tc)
            msg["tool_calls"] = calls
        u = resp.usage
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
        return msg, usage

    def complete(self, prompt: str) -> tuple[str, dict]:
        resp = self._create(messages=[{"role": "user", "content": prompt}])
        u = resp.usage
        return resp.choices[0].message.content or "", {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        }


class OracleClient:
    """Pipeline test client. Plays the OracleAgent through the normal loop."""

    model = "oracle"

    def __init__(self, env):
        from ehr.oracle import OracleAgent

        self.agent = OracleAgent(env)
        self.n = 0

    def chat(self, messages, tools):
        name, args = self.agent.next_call()
        self.n += 1
        msg = {"role": "assistant", "content": "",
               "tool_calls": [{"id": f"call_{self.n}", "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}}]}
        return msg, {"prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, prompt):
        return "Oracle summary of earlier work.", {"prompt_tokens": 0, "completion_tokens": 0}
