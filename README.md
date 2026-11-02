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

## Layout

```
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
