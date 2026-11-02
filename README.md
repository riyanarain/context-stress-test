# context-stress-test

CS 224V project by Brooke Ballhaus and Riya Narain.

We stress-test an automatic, lossless context-management harness for agents on long-horizon tasks, find where it breaks, and implement a task-agnostic improvement to its context policy. The harness keeps every tool result in full, shows the model a bounded working context that lists what is stored, and lets the agent's code read items back by reference. Base system: https://arxiv.org/abs/2602.07962

## Research question

Can agents scale to tasks with much more data and much longer horizons using automatic context management, and where does it fail?

## Testbeds

All testbeds share one task template: the agent gets a large collection of documents or records and must answer questions that need details from many of them over many dependent steps. Answers are checked automatically against gold labels.

- Biomedical: questions over clinical trial records (or an existing benchmark such as BixBench or MedAgentBench)
- Legal: questions over contracts with clause annotations (for example CUAD)
- Finance (optional): questions over SEC 10-K filings

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

## Tests

```
python -m pytest
```

## Layout

```
core/         shared task format, checker, document store
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
