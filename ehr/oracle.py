"""An oracle agent that reads the environment's hidden state to solve a task.

It is only for checking that every task is solvable and that the agent loop,
policies, and grading work end to end without an API key. It reads the
environment directly, so its score says nothing about model ability.
"""

from __future__ import annotations

from ehr.env import FORMULARY, EHREnv, ingredient
from ehr.grade import expected_actions


def _write_call(key: tuple, env: EHREnv) -> tuple[str, dict]:
    kind = key[0]
    if kind == "hold":
        return "hold_medication", {"patient_id": key[1], "medication": key[2], "reason": "per task rule"}
    if kind == "message":
        return "send_message", {"provider_id": key[1], "patient_id": key[2], "text": "Holding lisinopril for high potassium."}
    if kind == "schedule":
        return "schedule_appointment", {"patient_id": key[1], "appointment_type": key[2]}
    if kind == "order":
        drug = next(f for f in FORMULARY if ingredient(f) == key[2])
        return "order_medication", {"patient_id": key[1], "medication": drug, "reason": "per task rule"}
    raise ValueError(kind)


class OracleAgent:
    def __init__(self, env: EHREnv, idle_calls: int = 3):
        self.env = env
        self.queue: list[tuple[str, dict]] = []
        fam = env.task["family"]
        if fam == "update":
            self.queue = [("get_labs", {"patient_id": pid, "test_name": "potassium"}) for pid in sorted(env.patients)]
        elif fam == "interleave":
            self.queue = [("read_inbox", {})]
        self.idle_left = idle_calls

    def next_call(self) -> tuple[str, dict]:
        if self.queue:
            return self.queue.pop(0)
        env = self.env
        want = expected_actions(env)
        for a in env.actions:
            if a["active"] and tuple(a["key"]) not in want:
                return "cancel_action", {"action_id": a["id"]}
        have = env.active_actions()
        for key in sorted(want - have):
            return _write_call(key, env)
        if self.idle_left > 0:
            self.idle_left -= 1
            return ("read_inbox" if env.task["family"] == "interleave" else "list_patients"), {}
        return "finish", {"summary": "done"}
