"""The mock EHR environment: state, tools, timed events, and actions.

A task file (see ehr/generate.py) holds the full initial state. EHREnv makes a
private copy, exposes tools in OpenAI function-calling format, applies events
(new lab results or inbox messages that arrive mid-task), and records every
write action. Grading compares the active write actions at the end with what
the task's rule says they should be, given the final state (ehr/grade.py).
"""

from __future__ import annotations

import copy
import json
import re

from ehr.records import age_on

FORMULARY = [
    "amoxicillin 500 MG oral capsule",
    "azithromycin 250 MG oral tablet",
    "metformin 500 MG oral tablet",
    "atorvastatin 20 MG oral tablet",
    "ibuprofen 400 MG oral tablet",
    "lisinopril 10 MG oral tablet",
]

APPOINTMENT_TYPES = [
    "Diabetes follow-up",
    "Cardiology follow-up",
    "Hypertension follow-up",
    "Dermatology consult",
    "Physical therapy",
    "Annual wellness visit",
]


def ingredient(med_name: str) -> str:
    """Canonical ingredient: first word of the medication name, lowercase letters only."""
    first = med_name.strip().split()[0] if med_name.strip() else ""
    return re.sub(r"[^a-z]", "", first.lower())


def _pid(args: dict) -> str:
    return str(args.get("patient_id", "")).strip().upper()


TOOL_SPECS = [
    ("list_patients", "List every patient in the system with sex and age.", {}, []),
    ("get_patient", "Get demographics for one patient.", {"patient_id": {"type": "string"}}, ["patient_id"]),
    ("get_problem_list", "Get all recorded diagnoses for a patient, with dates and whether they resolved.",
     {"patient_id": {"type": "string"}}, ["patient_id"]),
    ("get_medications", "Get every medication record for a patient, newest first, with medication ID, status, and prescriber.",
     {"patient_id": {"type": "string"}}, ["patient_id"]),
    ("get_allergies", "Get the allergy list for a patient.", {"patient_id": {"type": "string"}}, ["patient_id"]),
    ("get_encounters", "Get every encounter for a patient, newest first, including provider and clinician notes.",
     {"patient_id": {"type": "string"}}, ["patient_id"]),
    ("get_labs", "Get lab results and vital signs for a patient, newest first. Optionally filter by test name (case-insensitive substring).",
     {"patient_id": {"type": "string"}, "test_name": {"type": "string"}}, ["patient_id"]),
    ("read_inbox", "Read every message in the clinician inbox. Messages not shown before are marked NEW.", {}, []),
    ("hold_medication", "Hold an active medication for a patient. Returns an action ID.",
     {"patient_id": {"type": "string"}, "medication": {"type": "string", "description": "Medication name, e.g. lisinopril"},
      "reason": {"type": "string"}}, ["patient_id", "medication"]),
    ("order_medication", "Order a medication from the formulary for a patient. Returns an action ID.",
     {"patient_id": {"type": "string"}, "medication": {"type": "string", "enum": FORMULARY},
      "reason": {"type": "string"}}, ["patient_id", "medication"]),
    ("schedule_appointment", "Schedule an appointment for a patient. Returns an action ID.",
     {"patient_id": {"type": "string"}, "appointment_type": {"type": "string", "enum": APPOINTMENT_TYPES},
      "date": {"type": "string", "description": "YYYY-MM-DD"}}, ["patient_id", "appointment_type"]),
    ("send_message", "Send a secure message to a clinician about a patient. Returns an action ID.",
     {"provider_id": {"type": "string"}, "patient_id": {"type": "string"}, "text": {"type": "string"}},
     ["provider_id", "patient_id", "text"]),
    ("cancel_action", "Cancel an action you took earlier, by action ID.", {"action_id": {"type": "string"}}, ["action_id"]),
    ("finish", "Call when the task is complete.", {"summary": {"type": "string"}}, ["summary"]),
]


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": req},
            },
        }
        for name, desc, props, req in TOOL_SPECS
    ]


class EHREnv:
    def __init__(self, task: dict):
        self.task = task
        self.today = task["today"]
        self.patients: dict[str, dict] = copy.deepcopy(task["patients"])
        self.providers: dict[str, str] = dict(task["providers"])
        self.inbox: list[dict] = copy.deepcopy(task.get("inbox", []))
        self.events: list[dict] = copy.deepcopy(task.get("events", []))
        for ev in self.events:
            ev.setdefault("fired", False)
            ev.setdefault("countdown", None)
        self.actions: list[dict] = []
        self.n_calls = 0
        self.n_writes = 0
        self.done = False
        self.call_log: list[dict] = []

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict | str | None) -> str:
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return "Error: arguments are not valid JSON."
        args = args or {}
        fn = getattr(self, f"_t_{name}", None)
        self.n_calls += 1
        if fn is None:
            out = f"Error: unknown tool {name}."
        else:
            try:
                out = fn(args)
            except Exception as e:  # tool errors are shown to the agent
                out = f"Error: {e}"
        self.call_log.append({"tool": name, "args": args})
        alerts = self._advance_events(name, args)
        if alerts:
            out += "\n\n" + "\n".join(alerts)
        return out

    def _patient(self, args: dict) -> dict:
        pid = _pid(args)
        if pid not in self.patients:
            raise ValueError(f"no patient with ID {pid!r}")
        return self.patients[pid]

    # ------------------------------------------------------------ read tools
    def _t_list_patients(self, args):
        lines = [f"{pid}  sex {p['sex']}  age {age_on(p['birth_date'], self.today)}"
                 for pid, p in sorted(self.patients.items())]
        return f"{len(lines)} patients:\n" + "\n".join(lines)

    def _t_get_patient(self, args):
        p = self._patient(args)
        return (f"Patient {p['id']}\nSex: {p['sex']}\nBirth date: {p['birth_date']}\n"
                f"Age: {age_on(p['birth_date'], self.today)}")

    def _t_get_problem_list(self, args):
        p = self._patient(args)
        rows = sorted(p["conditions"], key=lambda c: c["date"], reverse=True)
        lines = [f"{c['date']}  {c['description']}" + (f"  (resolved {c['resolved']})" if c["resolved"] else "  (active)")
                 for c in rows]
        return f"Problem list for {p['id']} ({len(lines)} entries):\n" + ("\n".join(lines) or "none")

    def _t_get_medications(self, args):
        p = self._patient(args)
        rows = sorted(p["medications"], key=lambda m: m["start"], reverse=True)
        lines = []
        for m in rows:
            status = "active" if not m["stop"] else f"stopped {m['stop']}"
            if any(a["type"] == "hold" and a["med_id"] == m["id"] and a["active"] for a in self.actions):
                status += ", ON HOLD"
            line = f"{m['id']}  {m['name']}  started {m['start']}  {status}  prescriber {m['prescriber']}"
            if m["reason"]:
                line += f"  reason: {m['reason']}"
            lines.append(line)
        return f"Medications for {p['id']} ({len(lines)} records):\n" + ("\n".join(lines) or "none")

    def _t_get_allergies(self, args):
        p = self._patient(args)
        lines = [f"{a['date']}  {a['substance']}" + (f"  reaction: {a['reaction']}" if a["reaction"] else "")
                 for a in p["allergies"]]
        return f"Allergy list for {p['id']}:\n" + ("\n".join(lines) or "no known allergies recorded")

    def _t_get_encounters(self, args):
        p = self._patient(args)
        rows = sorted(p["encounters"], key=lambda e: e["date"], reverse=True)
        lines = []
        for e in rows:
            line = f"{e['id']}  {e['date']}  {e['class']}  {e['description']}  provider {e['provider']}"
            if e["reason"]:
                line += f"  reason: {e['reason']}"
            lines.append(line)
            if e["note"]:
                lines.append(f"    Note: {e['note']}")
        return f"Encounters for {p['id']} ({len(rows)}):\n" + "\n".join(lines)

    def _t_get_labs(self, args):
        p = self._patient(args)
        flt = str(args.get("test_name") or "").strip().lower()
        rows = [o for o in p["observations"] if not flt or flt in o["name"].lower()]
        rows = sorted(rows, key=lambda o: o["date"], reverse=True)
        lines = [f"{o['date']}  {o['name']}  {o['value']} {o['units']}".rstrip() for o in rows]
        head = f"Labs for {p['id']}" + (f" matching {flt!r}" if flt else "") + f" ({len(lines)} results):\n"
        return head + ("\n".join(lines) or "none")

    def _t_read_inbox(self, args):
        shown = [m for m in self.inbox if m["delivered"]]
        if not shown:
            return "Inbox is empty."
        lines = []
        for m in shown:
            tag = "NEW  " if not m.get("seen") else ""
            lines.append(f"{tag}{m['id']}  {m['date']}  from {m['from']}: {m['text']}")
            m["seen"] = True
        return f"Inbox ({len(shown)} messages):\n" + "\n".join(lines)

    # ------------------------------------------------------------ write tools
    def _add_action(self, kind: str, key: list, **extra) -> str:
        self.n_writes += 1
        aid = f"A{len(self.actions) + 1:03d}"
        self.actions.append({"id": aid, "type": kind, "key": key, "active": True, **extra})
        return aid

    def _t_hold_medication(self, args):
        p = self._patient(args)
        want = str(args.get("medication", "")).strip().lower()
        if len(want) < 3:
            raise ValueError("medication name is required")
        for m in sorted(p["medications"], key=lambda m: m["start"], reverse=True):
            ing = ingredient(m["name"])
            if not m["stop"] and (ing and (ing in want or want in m["name"].lower())):
                aid = self._add_action("hold", ["hold", p["id"], ing], med_id=m["id"])
                return f"Held {m['name']} ({m['id']}) for {p['id']}. Action ID {aid}."
        raise ValueError(f"{p['id']} has no active medication matching {want!r}")

    def _t_order_medication(self, args):
        p = self._patient(args)
        med = str(args.get("medication", ""))
        match = next((f for f in FORMULARY if f.lower() == med.strip().lower()), None)
        if match is None:
            match = next((f for f in FORMULARY if ingredient(f) and ingredient(f) in med.lower()), None)
        if match is None:
            raise ValueError(f"{med!r} is not on the formulary: {FORMULARY}")
        aid = self._add_action("order", ["order", p["id"], ingredient(match)])
        return f"Ordered {match} for {p['id']}. Action ID {aid}."

    def _t_schedule_appointment(self, args):
        p = self._patient(args)
        typ = str(args.get("appointment_type", "")).strip()
        match = next((t for t in APPOINTMENT_TYPES if t.lower() == typ.lower()), None)
        if match is None:
            raise ValueError(f"appointment_type must be one of {APPOINTMENT_TYPES}")
        aid = self._add_action("schedule", ["schedule", p["id"], match], date=args.get("date", ""))
        return f"Scheduled {match} for {p['id']}. Action ID {aid}."

    def _t_send_message(self, args):
        p = self._patient(args)
        prov = str(args.get("provider_id", "")).strip().upper()
        if prov not in self.providers:
            raise ValueError(f"no provider with ID {prov!r}")
        aid = self._add_action("message", ["message", prov, p["id"]], text=args.get("text", ""))
        return f"Message sent to {prov} about {p['id']}. Action ID {aid}."

    def _t_cancel_action(self, args):
        aid = str(args.get("action_id", "")).strip().upper()
        for a in self.actions:
            if a["id"] == aid:
                if not a["active"]:
                    raise ValueError(f"action {aid} is already cancelled")
                a["active"] = False
                return f"Cancelled action {aid}."
        raise ValueError(f"no action with ID {aid!r}")

    def _t_finish(self, args):
        self.done = True
        return "Task marked complete."

    # ------------------------------------------------------------ events
    def _read_matches(self, trig: dict, name: str, args: dict) -> bool:
        if name != trig.get("tool") or _pid(args) != trig.get("patient"):
            return False
        flt = str(args.get("test_name") or "").strip().lower()
        target = trig.get("match", "").lower()
        return not flt or flt in target or target in flt

    def _advance_events(self, name: str, args: dict) -> list[str]:
        alerts = []
        is_write = name in {"hold_medication", "order_medication", "schedule_appointment", "send_message", "cancel_action"}
        for ev in self.events:
            if ev["fired"]:
                continue
            trig = ev["trigger"]
            if trig["type"] == "after_read":
                if ev["countdown"] is None:
                    if self._read_matches(trig, name, args):
                        ev["countdown"] = trig.get("delay", 2)
                    continue
                ev["countdown"] -= 1
                ready = ev["countdown"] <= 0
            elif trig["type"] == "after_writes":
                ready = is_write and self.n_writes >= trig["count"]
            else:
                ready = False
            if ready:
                self._apply(ev["effect"])
                ev["fired"] = True
                ev["fired_at_call"] = self.n_calls
                alerts.append(ev["alert"])
        return alerts

    def _apply(self, eff: dict) -> None:
        if eff["type"] == "add_observation":
            self.patients[eff["patient"]]["observations"].append(dict(eff["observation"]))
        elif eff["type"] == "deliver_messages":
            for m in self.inbox:
                if m["id"] in eff["ids"]:
                    m["delivered"] = True
        else:
            raise ValueError(eff["type"])

    # ------------------------------------------------------------ results
    def active_actions(self) -> set[tuple]:
        return {tuple(a["key"]) for a in self.actions if a["active"]}

    def pending_events(self) -> int:
        return sum(1 for ev in self.events if not ev["fired"])
