# AGENTS.md

Guide for coding agents (and people) working in this repo. Read this first.

## What this project is

CS 224V project (Brooke Ballhaus, Riya Narain; mentor Vidhyakshaya Kannan). We study what makes context hard for an agent to manage, independent of task length. The main testbed is an agentic mock EHR built on Synthea synthetic patients. The agent works through tools (read labs, medications, encounters, inbox; hold or order medications, schedule visits, message clinicians) and is graded on the final state of the system. Each task family varies one difficulty factor at three levels while the environment size stays about the same. We compare context management policies across these factors, find where they break, and then design a fix.

All patient data is synthetic. Do not add real patient data to this repo.

## Layout

```
ehr/                  agentic mock EHR (main testbed)
  records.py          loads Synthea CSVs into structured patient records; TODAY = 2021-12-01
  env.py              EHREnv: tools, state, mid-task events, write actions
  families.py         the 4 task families and their 3 difficulty levels
  grade.py            task rules and final-state grading
  generate.py         CLI: build task files
  oracle.py           cheating solver that reads hidden state (pipeline checks only)
  difficulty_check.py scripted strategies showing each difficulty knob works
agents/
  llm.py              OpenAI-compatible client, and OracleClient for offline runs
  policies.py         context policies: full, truncate, summary
  loop.py             runs one episode, writes a full trace
  run.py              CLI: run tasks x policies, resumable
  report.py           CLI: summary table by family, level, policy
core/                 shared helpers (set scoring, older QA task format)
testbeds/             older retrieval-style testbeds (CUAD legal, Synthea QA). Not agentic; kept for reference.
tests/                pytest suite; runs without data or API keys
data/                 generated data (gitignored)
runs/                 run outputs and traces (gitignored)
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash testbeds/biomedical/download.sh          # Synthea CSVs into data/raw/synthea/csv (about 60 MB zip)
python -m ehr.generate                         # 60 tasks into data/ehr/tasks
python -m pytest                               # should pass
python -m ehr.difficulty_check                 # sanity check of the difficulty knobs
python -m agents.run --client oracle --out runs/oracle-check   # pipeline check, no API key
```

Generation is deterministic for a given `--seed`, so everyone gets identical tasks. Do not commit `data/` or `runs/`.

## API keys

Put credentials in `.env` in the repo root (gitignored; see `.env.example`):

```
LLM_API_KEY=...
LLM_BASE_URL=https://openrouter.ai/api/v1     # omit for OpenAI itself
LLM_MODEL=...
```

Any OpenAI-compatible chat completions endpoint with tool calling works: OpenAI, OpenRouter, Groq, Gemini's OpenAI-compatible endpoint, or Anthropic's OpenAI SDK compatibility endpoint. Never commit keys. Never print keys in logs or traces.

## Running experiments

Start small. Each episode can use hundreds of thousands of prompt tokens, because the whole history is resent on every model call.

```
# pilot: 1 task per cell, easy vs hard, all policies
python -m agents.run --levels 0 2 --limit 1 --out runs/pilot --price-in 0.15 --price-out 0.6
python -m agents.report runs/pilot

# full grid
python -m agents.run --out runs/main --workers 4
python -m agents.report runs/main --csv
```

`--price-in` and `--price-out` are USD per million tokens for your model and only affect the cost column. Reruns skip finished episodes, so an interrupted run can be resumed with the same command. Use `--temperature -1` for models that reject a temperature setting.

Useful flags: `--families`, `--levels`, `--limit` (tasks per family and level), `--policies`, `--max-steps` (default 80), `--truncate-chars` (default 3000), `--summary-budget` (tokens, default 16000).

## Task families

| Family | Rule the agent must follow | Difficulty factor | L0 | L1 | L2 |
|---|---|---|---|---|---|
| update | Hold lisinopril and message its prescriber if the latest potassium is above 5.0 | Facts that change | one value | several values, earlier ones on the other side of 5.0 | as L1, plus new results posted mid-task that flip two patients |
| distractor | Schedule a Diabetes follow-up if latest A1c is above 6.5 and the last encounter was over 182 days ago | Similar distractors | far from thresholds | closer | near-misses (A1c 6.4 or 6.5, 176 to 182 days) |
| distance | Order amoxicillin for strep, or azithromycin if there is a penicillin allergy | Where the needed fact sits | allergy list | note 6 to 18 months back | note 8 to 15 years back, deep in the encounter list |
| interleave | Carry out inbox requests; later messages change or cancel earlier ones | Interleaved goals | 5 independent requests | plus 2 amendments | amendments arrive after the 3rd write action |

Default tasks have 8 patients and an environment of about 55K to 75K tokens. Levels within a family use the same patients (same seed and index), so only the difficulty factor changes.

Grading (`ehr/grade.py`) recomputes the expected actions from the final environment state, so mid-task events count only if they fired. Scores are set-based: exact match, precision, recall, F1 over normalized actions such as `hold|P0012|lisinopril` or `message|DR003|P0012`.

## Outputs

`runs/<name>/results.jsonl` has one line per episode: task, family, level, policy, model, exact, f1, precision, recall, missing and extra actions, steps, prompt and completion tokens, max context, error, cost.

`runs/<name>/traces/<task_id>__<policy>.jsonl` has one record per event:
- `model`: step, estimated context tokens, usage, content, tool calls
- `tool`: name, args, full output, output_chars, shown_chars (after the policy), policy_event if truncated
- `policy`: summarize events, with tokens before and after and the summary text
- `error`, and a final `result`

The trace audit should work from these files. Full tool outputs are stored even when the policy hid part of them, so you can check whether a needed fact was ever shown to the model.

## Adding a context policy

Subclass `FullContext` in `agents/policies.py`. Override `process_tool_output(text)` to change outputs before they enter the history, and/or `before_call(messages, client)` to rewrite the history before each model call (return a list of event dicts for the trace). Register it in `make_policy` and `POLICIES`. Keep tool messages directly after the assistant message that called them, or the API will reject the request.

## Adding a task family

Add a builder in `ehr/families.py` (use `select_patients` and `finalize`), its rule in `ehr/grade.py` (`RULES`), instructions in `INSTRUCTIONS`, oracle support in `ehr/oracle.py` if it needs reads to trigger events, and a scripted strategy in `ehr/difficulty_check.py` that shows the knob works. Then add tests.

## Conventions

- Run `python -m pytest` before committing. Tests must not need data downloads or API keys.
- Keep tasks deterministic: all randomness goes through the seeded `random.Random` passed to builders.
- Do not add AI tools as commit authors or co-authors (no Co-Authored-By trailers).
- Do not commit the mentor lab's paper (under submission) or their Codex library build. Reference them only in private notes.
- Plain, short prose in docs. No em dashes.
