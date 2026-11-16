"""Parse CHI-Bench trial folders.

A trial folder (from `cb experiment run` logs, or a leaderboard packet) has:
    result.json                      task, agent, model, token usage, cost
    verifier/reward.json             {"reward": 0.0 | 1.0}
    verifier/scorecard.json          per-check verdicts
    agent/trajectory.jsonl.zst       ATIF trajectory, one JSON object per line
                                     (or agent/trajectory.jsonl / trajectory.json)

ATIF lines: an optional {"_atif_header": {...}} line, then steps with
step_id, source (system | user | agent), message, tool_calls
[{tool_call_id, function_name, arguments}], observation {results: [{source_call_id,
content}]}, and for some harnesses metrics {prompt_tokens, completion_tokens, ...}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolCall:
    step: int
    call_id: str
    name: str
    args_text: str
    output: str = ""


@dataclass
class Trial:
    path: Path
    header: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    scorecard: dict = field(default_factory=dict)
    reward: float | None = None

    @property
    def calls(self) -> list[ToolCall]:
        out = []
        for s in self.steps:
            outputs = {}
            for r in (s.get("observation") or {}).get("results") or []:
                outputs[r.get("source_call_id")] = _text(r.get("content"))
            for tc in s.get("tool_calls") or []:
                args = tc.get("arguments")
                args_text = args if isinstance(args, str) else json.dumps(args)
                cid = tc.get("tool_call_id", "")
                out.append(ToolCall(s.get("step_id", 0), cid, tc.get("function_name", ""), args_text,
                                    outputs.get(cid, "")))
        return out


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content)


def read_jsonl_zst(path: Path) -> list[dict]:
    import zstandard

    with path.open("rb") as f:
        raw = zstandard.ZstdDecompressor().stream_reader(f).read().decode("utf-8", "replace")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_trajectory(agent_dir: Path) -> list[dict]:
    for name in ("trajectory.jsonl.zst", "trajectory.jsonl", "trajectory.json"):
        p = agent_dir / name
        if not p.exists():
            continue
        if name.endswith(".zst"):
            return read_jsonl_zst(p)
        text = p.read_text()
        if name.endswith(".jsonl"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, dict) and "steps" in data:  # single-document ATIF
            return [{"_atif_header": {k: v for k, v in data.items() if k != "steps"}}] + data["steps"]
        return data
    return []


def load_trial(path: str | Path) -> Trial:
    path = Path(path)
    lines = load_trajectory(path / "agent")
    header = {}
    steps = []
    for obj in lines:
        if "_atif_header" in obj:
            header = obj["_atif_header"]
        else:
            steps.append(obj)
    t = Trial(path=path, header=header, steps=steps)
    if (path / "result.json").exists():
        t.result = json.loads((path / "result.json").read_text())
    sc = path / "verifier" / "scorecard.json"
    if sc.exists():
        t.scorecard = json.loads(sc.read_text())
    rw = path / "verifier" / "reward.json"
    if rw.exists():
        t.reward = json.loads(rw.read_text()).get("reward")
    elif t.scorecard:
        t.reward = t.scorecard.get("binary_reward")
    return t


def find_trials(root: str | Path) -> list[Path]:
    """Every folder under root that has a result.json and an agent/ trajectory."""
    out = []
    for r in Path(root).rglob("result.json"):
        d = r.parent
        if (d / "agent").is_dir():
            out.append(d)
    return sorted(out)
