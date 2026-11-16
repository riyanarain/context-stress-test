import json
from pathlib import Path

import zstandard

from chibench.analyze import build_report, drop_duplicate_submissions, load, stratified_diff
from chibench.audit import audit, write_csv
from chibench.trajectory import find_trials, load_trial

HB = "/workspace/skills/managed-care-operations-handbook/references/payer-um/01-workflow.md"


def make_trial(root: Path, sub: str, domain: str, task: str, *, reward: float, agent: str, steps: list[dict],
               cost=1.0, tokens=1000):
    d = root / sub / "trials" / domain / f"{task}__x{len(list((root / sub).rglob('result.json'))) if (root / sub).exists() else 0}"
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    lines = [{"_atif_header": {"agent": {"name": agent, "model_name": "m"}}}] + steps
    raw = "\n".join(json.dumps(x) for x in lines).encode()
    (d / "agent" / "trajectory.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(raw))
    (d / "result.json").write_text(json.dumps({
        "task_name": f"actava-ai/{task}",
        "agent_info": {"name": agent, "model_info": {"name": "model-x"}},
        "agent_result": {"n_input_tokens": tokens, "n_cache_tokens": 0, "n_output_tokens": 10, "cost_usd": cost},
    }))
    (d / "verifier" / "reward.json").write_text(json.dumps({"reward": reward}))
    (d / "verifier" / "scorecard.json").write_text(json.dumps({
        "binary_reward": reward, "fractional_reward": 1.0 if reward else 0.8, "total_checks": 5,
        "check_scores": {"outcome.target_status": reward, "judge.md_review:final_decision": 1.0, "md.audit": None},
    }))
    return d


def codex_steps(truncated: bool):
    out = "Output:\nfine"
    if truncated:
        out = "Warning: truncated output (original token count: 13046)\nTotal output lines: 657"
    return [
        {"step_id": 1, "source": "system", "message": "sys"},
        {"step_id": 2, "source": "agent", "message": "x",
         "tool_calls": [{"tool_call_id": "c1", "function_name": "exec",
                         "arguments": {"input": f"tools.exec_command({{cmd:'sed -n 1,300p {HB}'}}); "
                                                "await tools.review__save_md_review({})"}}],
         "observation": {"results": [{"source_call_id": "c1", "content": out}]}},
        {"step_id": 3, "source": "agent", "message": "x",
         "tool_calls": [{"tool_call_id": "c2", "function_name": "exec", "arguments": {"input": f"cat {HB}"}}],
         "observation": {"results": [{"source_call_id": "c2", "content": "ok"}]}},
    ]


def claude_steps(offload: bool, reread: bool):
    path = "/logs/agent/sessions/tool-results/toolu_9.txt"
    big = f"<persisted-output>\nOutput too large (55.8KB). Full output saved to: {path}\nPreview..." if offload else "small"
    steps = [
        {"step_id": 1, "source": "agent", "message": "x", "metrics": {"prompt_tokens": 20000},
         "tool_calls": [{"tool_call_id": "t1", "function_name": "mcp__chi-bench__payer_intake_hub_list_intake_queue",
                         "arguments": {}}],
         "observation": {"results": [{"source_call_id": "t1", "content": big}]}},
        {"step_id": 2, "source": "agent", "message": "x", "metrics": {"prompt_tokens": 90000},
         "tool_calls": [{"tool_call_id": "t2", "function_name": "Read", "arguments": {"file_path": HB}}],
         "observation": {"results": [{"source_call_id": "t2", "content": "doc"}]}},
    ]
    if reread:
        steps.append({"step_id": 3, "source": "agent", "message": "x", "metrics": {"prompt_tokens": 95000},
                      "tool_calls": [{"tool_call_id": "t3", "function_name": "Read", "arguments": {"file_path": path}}],
                      "observation": {"results": [{"source_call_id": "t3", "content": "full"}]}})
    return steps


def test_features_codex_and_claude(tmp_path):
    root = tmp_path / "subs"
    make_trial(root, "codexsub", "pa_um", "pa_t001", reward=0.0, agent="codex", steps=codex_steps(True))
    make_trial(root, "ccsub", "pa_um", "pa_t001", reward=1.0, agent="claude-code", steps=claude_steps(True, False))
    rows = {r["submission"]: r for r in audit(root)}
    c = rows["codexsub"]
    assert c["truncation_events"] == 1 and c["tokens_hidden"] == 13046
    assert c["handbook_reads"] == 2 and c["handbook_docs"] == 1 and c["handbook_rereads"] == 1
    assert c["n_app_calls"] == 1  # the review__save_md_review call embedded in the exec script
    assert c["reward"] == 0.0 and c["failed_checks"] == "outcome.target_status"
    assert c["domain"] == "pa_um" and c["task"] == "pa_t001" and c["n_steps"] == 2
    k = rows["ccsub"]
    assert k["offload_events"] == 1 and k["offload_never_read"] == 1
    assert k["peak_prompt_tokens"] == 90000 and k["n_app_calls"] == 1


def test_offloaded_output_that_is_reopened(tmp_path):
    root = tmp_path / "subs"
    make_trial(root, "cc", "cm", "cm_x", reward=1.0, agent="claude-code", steps=claude_steps(True, True))
    (row,) = audit(root)
    assert row["offload_events"] == 1 and row["offload_never_read"] == 0


def test_load_trial_reads_plain_jsonl(tmp_path):
    d = tmp_path / "t"
    (d / "agent").mkdir(parents=True)
    (d / "agent" / "trajectory.jsonl").write_text("\n".join(json.dumps(s) for s in codex_steps(False)))
    (d / "result.json").write_text("{}")
    t = load_trial(d)
    assert len(t.calls) == 2 and t.calls[0].output.startswith("Output")
    assert find_trials(tmp_path) == [d]


def test_analysis_report_and_stratified_diff(tmp_path):
    root = tmp_path / "subs"
    for i, task in enumerate(["pa_t001", "pa_t002", "pa_t003"]):
        make_trial(root, "codexsub", "pa_um", task, reward=0.0, agent="codex", steps=codex_steps(True), tokens=500 + i)
        make_trial(root, "ccsub", "pa_um", task, reward=1.0, agent="claude-code", steps=claude_steps(False, False),
                   tokens=900 + i)
    # an exact re-upload of ccsub should be dropped
    make_trial(root, "ccsub-copy", "pa_um", "pa_t001", reward=1.0, agent="claude-code", steps=claude_steps(False, False), tokens=900)
    make_trial(root, "ccsub-copy", "pa_um", "pa_t002", reward=1.0, agent="claude-code", steps=claude_steps(False, False), tokens=901)
    make_trial(root, "ccsub-copy", "pa_um", "pa_t003", reward=1.0, agent="claude-code", steps=claude_steps(False, False), tokens=902)
    write_csv(audit(root), tmp_path / "a.csv")
    rows, dropped = drop_duplicate_submissions(load(tmp_path / "a.csv"))
    assert len(dropped) == 1 and dropped[0] in ("ccsub", "ccsub-copy")  # keeps one copy (the later name)
    d = stratified_diff(rows, lambda r: (r["truncation_events"] or 0) > 0)
    assert d["diff"] == -1.0 and d["tasks"] == 3
    report = build_report(rows, dropped)
    assert "Per submission" in report and "codexsub" in report and "within task and harness" in report


def test_configs_are_well_formed():
    import yaml

    for p in Path("chibench/configs").glob("*.yaml"):
        cfg = yaml.safe_load(p.read_text())
        assert cfg["rows"] and set(cfg["domains"]) <= {"pa_provider", "pa_um", "cm"}
        assert cfg["defaults"]["trials_root"].startswith("logs/experiments/ctx_")
