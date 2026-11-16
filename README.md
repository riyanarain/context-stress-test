# context-stress-test

CS 224V project by Brooke Ballhaus and Riya Narain. Mentor: Vidhyakshaya Kannan.

Agents can only use what is in their context, and context management policies such as truncation, summarization, and compaction each lose information in different ways. These losses matter most in long-horizon work. We study this on CHI-Bench, a benchmark of long-horizon healthcare workflows across 20 simulated apps and a 1,279-document policy handbook. In its Marathon setting, where 25 tasks run in one session, the best reported agent drops from 28% to 3.8%. We compare production Codex with Codex using our mentor lab's context management release, audit traces to separate context failures from planning failures, and design a fix for the most common one.

See [AGENTS.md](AGENTS.md) for setup, commands, and repo details.

## Main testbed: CHI-Bench

- `chibench/` has setup and run scripts, three run configs (a 3-task smoke test, one 25-task domain, one Marathon session), a cost estimator, and a trace audit that works on any CHI-Bench trial folder.
- [reports/chibench_public.md](reports/chibench_public.md) audits the 1,375 public leaderboard trials. Within the same task and harness, trials that loaded more context passed more often, and how much Codex truncated did not predict failure. Single tasks peak around 120K to 230K prompt tokens. This suggests context is not yet the bottleneck in single tasks, which is why we focus on Marathon sessions.

## Sandbox: agentic mock EHR

Our first agentic testbed, now a free sandbox for testing ideas: an electronic health record system built on Synthea synthetic patients (no real patient data). The agent works through tools: reading labs, medications, encounters, allergies, and a clinician inbox, and holding or ordering medications, scheduling appointments, messaging clinicians, and cancelling earlier actions. It is graded on the final state of the system.

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
