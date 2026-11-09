# context-stress-test

CS 224V project by Brooke Ballhaus and Riya Narain. Mentor: Vidhyakshaya Kannan.

Agents can only use what is in their context, and context management policies such as truncation, summarization, and compaction each lose information in different ways. Existing evaluations mostly scale task length, but length does not necessarily reflect how hard a task's context is to manage. We study which properties of a task make context hard to manage, independent of length, compare policies across those properties, and then design a fix for the gap that matters most.

See [AGENTS.md](AGENTS.md) for setup, commands, and repo details.

## Main testbed: agentic mock EHR

An electronic health record system built on Synthea synthetic patients (no real patient data). The agent works through tools: reading labs, medications, encounters, allergies, and a clinician inbox, and holding or ordering medications, scheduling appointments, messaging clinicians, and cancelling earlier actions. It is graded on the final state of the system.

Four task families each vary one difficulty factor at three levels, with the environment size held about constant (8 patients, about 55K to 75K tokens):

| Family | Difficulty factor | Hard level |
|---|---|---|
| update | Facts that change over time | Earlier potassium values contradict the latest, and new results arrive mid-task |
| distractor | Similar distractors | Near-miss A1c values and visit dates right at the thresholds |
| distance | Where a needed fact sits | Penicillin allergy only in a clinician note 8 to 15 years back |
| interleave | Interleaved goals | Amendments to earlier inbox requests arrive mid-task |

Scripted strategies with one known weakness each confirm the knobs work. Each strategy is fully correct at level 0 and fails at the level that targets its weakness (`python -m ehr.difficulty_check`):

| Family | Strategy | L0 | L1 | L2 |
|---|---|---|---|---|
| update | uses the earliest value | 1.00 | 0.00 | 0.00 |
| update | never rechecks after acting | 1.00 | 1.00 | 0.00 |
| distractor | off-by-one thresholds | 1.00 | 1.00 | 0.00 |
| distance | reads only the first 3000 characters | 1.00 | 1.00 | 0.40 |
| interleave | ignores amendments | 1.00 | 0.00 | 0.00 |
| interleave | reads the inbox once | 1.00 | 1.00 | 0.00 |

## Context policies compared

- full: keep the whole history
- truncate: cap each tool output, keeping head and tail
- summary: replace older turns with a model-written summary when the history passes a token budget
- planned: production Codex compaction and the lab's context-management release, if feasible

## Quick start

```
pip install -r requirements.txt
bash testbeds/biomedical/download.sh
python -m ehr.generate
python -m agents.run --client oracle --out runs/oracle-check     # no API key needed
cp .env.example .env                                            # add LLM_API_KEY, LLM_MODEL
python -m agents.run --levels 0 2 --limit 1 --out runs/pilot
python -m agents.report runs/pilot
```

## Earlier testbeds

`testbeds/legal` (CUAD contracts) and `testbeds/biomedical` (Synthea charts) are retrieval-style question answering over document collections, built in week 6. They are not agentic, so they are no longer the main testbed, but the Synthea download script is reused. Related benchmark: LOCA-bench (https://github.com/hkust-nlp/LOCA-bench).

## Tests

```
python -m pytest
```
