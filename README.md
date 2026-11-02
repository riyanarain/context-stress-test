# context-stress-test

CS 224V project by Brooke Ballhaus and Riya Narain.

We stress-test an automatic, lossless context-management harness for agents on long-horizon tasks, find where it breaks, and implement a task-agnostic improvement to its context policy. The harness keeps every tool result in full, shows the model a bounded working context that lists what is stored, and lets the agent's code read items back by reference.

The original benchmark is LOCA-bench (https://github.com/hkust-nlp/LOCA-bench, paper: https://arxiv.org/abs/2602.07962), which scales mock-service tasks to fixed context sizes and ships several context strategies we use as baselines. This repo adds our own domain testbeds in the same spirit.

## Research question

Can agents scale to tasks with much more data and much longer horizons using automatic context management, and where does it fail?

## Testbeds

All testbeds share one task template: the agent gets a large collection of documents or records and must answer questions that need details from many of them over many dependent steps. Answers are checked automatically against gold labels.

- Legal: questions over lawyer-annotated contracts (CUAD). Built.
- Biomedical: questions over synthetic patient charts (Synthea). Built.
- Finance (optional): questions over SEC 10-K filings. Not started.

## Scaling knobs

- Collection size: number of documents or records
- Horizon: number of dependent steps
- Distractors: irrelevant data mixed into the collection

## Baselines

- Full context with a long window
- Head and tail truncation of tool outputs
- Rolling summarization of older turns
- Production agent with default compaction
- The harness (system under test)

We measure accuracy, tokens, and dollar cost.

## Task format

Every testbed produces tasks in the same format (`core/task.py`). A task gives the agent a collection of documents and asks which ones satisfy a chain of conditions. The answer is a set of document IDs, scored automatically with exact match, precision, recall, and F1 (`core/checker.py`). Agents end their answer with a line like `ANSWER: C0001, C0002`.

Each task can be run two ways. In single-turn mode the agent gets all conditions in one question. In multi-turn mode it gets one condition per step, and each step narrows the previous answer, so the number of steps is the horizon knob and per-step scores show where errors start.

`core/docstore.py` builds a workspace per task with only that task's documents in `docs/` and a `task.json` with the questions (no gold answers).

## Legal testbed (CUAD)

Built on CUAD v1 (The Atticus Project, CC BY 4.0): 510 commercial contracts annotated by lawyers for 41 clause types. Conditions are a clause being present or absent (21 clause types) or the contract being governed or not governed by a given jurisdiction. Every step strictly shrinks the candidate set and leaves at least one contract. Gold answers come from the CUAD annotations. Seven contracts whose governing law is redacted are left out.

```
bash testbeds/legal/download.sh                      # CUAD into data/raw/cuad
python -m testbeds.legal.generate --n-docs 10 50 200 --n-steps 1 3 5 --n-tasks 10
```

This writes all contracts to `data/legal/docs/` and 90 tasks to `data/legal/tasks.jsonl`. Contracts are long, so collections are large:

| Collection size | Approximate tokens |
|---|---|
| 10 contracts | 100K |
| 50 contracts | 650K |
| 200 contracts | 2.6M |

For cheap debugging, generate a few tasks with `--n-docs 3 --n-tasks 2`.

## Biomedical testbed (Synthea)

Built on the Synthea synthetic patient sample (CSV release, Nov 2021): about 1,160 fully synthetic patients, so there is no real patient data. Each patient becomes one text chart with a short header and a dated log of encounters, diagnoses, medications, observations, procedures, immunizations, allergies, and care plans. Names, addresses, and ID numbers are left out.

Conditions are a recorded diagnosis (20 clinical diagnoses), a prescribed medication ingredient (13), the most recent value of a lab or vital above a threshold (BMI, systolic blood pressure, hemoglobin A1c, total and LDL cholesterol), sex, birth year, and death date. Lab conditions need the latest value across the whole chart, so the agent has to track details spread over many encounters. Patients with ambiguous latest lab values and 13 charts over 400K characters are left out.

```
bash testbeds/biomedical/download.sh                 # Synthea CSV into data/raw/synthea/csv
python -m testbeds.biomedical.generate --n-docs 10 50 200 --n-steps 1 3 5 --n-tasks 10
python -m testbeds.biomedical.verify                 # re-checks every gold answer from chart text only
```

Collection sizes are similar to the legal testbed: about 100K tokens for 10 patients, 550K for 50, and 2.3M for 200.

## Tests

```
python -m pytest
```

## Layout

```
core/         shared task format, checker, document store, chained-condition generator
tests/        unit tests
harness/      wrapper around the prototype
baselines/    truncation, rolling summary, production agent
testbeds/     one folder per domain: data loader and question generator
scaling/      scripts that vary collection size and step count
traces/       logged runs (large files are gitignored)
audit/        trace analysis and failure labeling
results/      plots and tables
```

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # add API keys here, never commit .env
```
