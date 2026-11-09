import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agents.llm import OpenAIChatClient, OracleClient
from agents.loop import run_episode
from agents.policies import RollingSummary, Truncate, est_tokens, make_policy
from tests.test_ehr import build


def test_truncate_keeps_head_and_tail():
    p = Truncate(max_chars=100)
    text = "A" * 200 + "B" * 200
    out, ev = p.process_tool_output(text)
    assert out.startswith("A") and out.endswith("B") and "omitted" in out
    assert ev["original_chars"] == 400
    assert p.process_tool_output("short") == ("short", None)


class FakeSummarizer:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return "SUMMARY TEXT", {"prompt_tokens": 10, "completion_tokens": 5}


def fake_history(n_groups, out_chars=2000):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "the task"}]
    for i in range(n_groups):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "get_labs", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * out_chars})
    return msgs


def test_summary_replaces_old_groups_and_keeps_pairs_valid():
    pol = RollingSummary(budget_tokens=3000, keep_recent=3)
    msgs = fake_history(20)
    client = FakeSummarizer()
    events = pol.before_call(msgs, client)
    assert events and events[0]["event"] == "summarize"
    assert msgs[2]["content"].startswith("[Summary of earlier work]")
    assert sum(1 for m in msgs if m["role"] == "assistant") == 3
    # every tool message still follows its assistant tool call
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            prev = msgs[i - 1]
            assert prev["role"] == "assistant" and prev["tool_calls"][0]["id"] == m["tool_call_id"]
    # a second summary folds in the first
    msgs += fake_history(10)[2:]
    pol.before_call(msgs, client)
    assert "SUMMARY TEXT" in client.prompts[-1]
    assert est_tokens(msgs) < est_tokens(fake_history(30))


def test_summary_noop_under_budget():
    msgs = fake_history(2, out_chars=10)
    assert RollingSummary(budget_tokens=10_000).before_call(msgs, FakeSummarizer()) == []


def test_episode_with_oracle_all_policies(tmp_path):
    task = build("update", 2)
    for name in ("full", "truncate", "summary"):
        res = run_episode(task, None, make_policy(name), trace_path=tmp_path / f"{name}.jsonl",
                          client_factory=OracleClient)
        assert res["exact"] == 1.0 and res["error"] is None
        lines = [json.loads(x) for x in (tmp_path / f"{name}.jsonl").read_text().splitlines()]
        assert lines[-1]["type"] == "result"
        assert any(r["type"] == "tool" and "[ALERT]" in r["output"] for r in lines)


class _Handler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.calls.append(body)
        n = len(_Handler.calls)
        if "tools" in body:
            name = "list_patients" if n == 1 else "finish"
            args = "{}" if n == 1 else json.dumps({"summary": "done"})
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": args}}]}
        else:
            msg = {"role": "assistant", "content": "a summary"}
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_openai_client_against_local_server(tmp_path, monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = OpenAIChatClient(model="fake-model", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                  api_key="test")
        res = run_episode(build("distance", 0), client, make_policy("full"), trace_path=tmp_path / "t.jsonl")
        assert res["error"] is None and res["finished"] and res["model_calls"] == 2
        assert res["prompt_tokens"] == 200
        sent = _Handler.calls[1]["messages"]
        assert sent[2]["tool_calls"][0]["function"]["name"] == "list_patients"
        assert sent[3]["role"] == "tool" and "patients" in sent[3]["content"]
        assert client.complete("hi")[0] == "a summary"
    finally:
        srv.shutdown()
