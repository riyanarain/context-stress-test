# CHI-Bench public trajectories: context audit

1375 trials from 19 leaderboard submissions (cm, pa_provider, pa_um). Duplicate packets dropped: none.

## Per submission

| submission | agent | pass@1 | mean partial credit | median tool calls | median input tokens | median peak prompt | trials with truncation | trials with offload | median handbook docs | mean cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-09-pa-codex-agent-gpt-5-5 | codex | 0.68 | 0.88 | 35 | 1,536,762 | n/a | 0% | 0% | 6 | $1.85 |
| 2026-07-26-erius-opus5 | claude-code | 0.55 | 0.93 | 40 | 3,991,623 | 229,998 | 0% | 44% | 5 | $5.44 |
| 2026-06-05-erius | claude-code | 0.37 | 0.82 | 42 | 3,357,640 | 215,250 | 0% | 36% | 7 | $4.76 |
| 2026-07-24-opus-5-claude-code | claude-code | 0.37 | 0.86 | 46 | 4,666,373 | 234,108 | 0% | 61% | 7 | $5.07 |
| 2026-09-23-u2med-hermes-deepseek-v4-pro-v6-all-u2 | U2MedFellow v6 | 0.37 | 0.82 | 26 | 0 | n/a | 0% | 0% | 4 | n/a |
| 2026-05-28-claude-opus-4-8 | claude-code | 0.33 | 0.88 | 43 | 2,834,250 | 169,474 | 0% | 37% | 7 | $3.31 |
| chi-bench-leaderboard-2026-05-15-claude-code-anthropic-claude-opus-4-6 | claude-code | 0.28 | 0.81 | 39 | 2,138,165 | 132,140 | 0% | 53% | 7 | $2.35 |
| 2026-10-06-hg-chi-full75-20261006 | codex | 0.27 | 0.80 | 38 | 3,172,750 | 121,166 | 100% | 0% | 8 | $0.33 |
| 2026-07-24-gpt-5-6-sol-codex | codex | 0.25 | 0.83 | 30 | 2,005,227 | n/a | 99% | 0% | 7 | $2.06 |
| 2026-07-24-kimi-k3-openai-agents | openai-agents | 0.25 | 0.83 | 41 | 2,100,901 | 116,677 | 27% | 0% | 7 | n/a |
| 2026-07-22-fable-5-claude-code | claude-code | 0.24 | 0.85 | 41 | 3,942,509 | 191,206 | 0% | 57% | 7 | $7.38 |
| 2026-07-06-cuilinke-hermes-medguard | hermes | 0.23 | 0.55 | 27 | 0 | n/a | 12% | 0% | 3 | n/a |
| 2026-07-06-sonnet-5-claude-code | claude-code | 0.20 | 0.80 | 45 | 3,775,868 | 184,601 | 0% | 32% | 7 | $2.29 |
| 2026-06-02-erius | claude-code | 0.19 | 0.59 | 37 | 2,594,547 | 172,700 | 0% | 37% | 7 | $3.18 |
| 2026-07-06-glm-5-2-openai-agents | openai-agents | 0.19 | 0.69 | 37 | 2,085,028 | 120,420 | 25% | 0% | 6 | n/a |
| 2026-07-24-gpt-5-6-luna-codex | codex | 0.13 | 0.77 | 29 | 2,066,745 | n/a | 100% | 0% | 7 | $0.44 |
| 2026-07-24-gpt-5-6-terra-codex | codex | 0.13 | 0.79 | 25 | 1,477,260 | n/a | 100% | 0% | 7 | $0.84 |
| 2026-07-24-inkling-256k-openai-agents | openai-agents | 0.08 | 0.32 | 13 | 349,505 | 60,456 | 15% | 0% | 5 | n/a |
| 2026-07-24-nemotron-3-ultra-openai-agents | openai-agents | 0.00 | 0.22 | 8 | 342,825 | 49,140 | 4% | 0% | 0 | n/a |

## Context signals vs passing, within the same task

Positive means trials with the signal pass more often than trials of the same task without it. 95% intervals come from resampling tasks.

| signal | pass-rate difference | 95% interval | tasks | trials with | trials without |
|---|---|---|---|---|---|
| any tool output truncated | -0.077 | [-0.132, -0.029] | 75 | 361 | 1014 |
| any output offloaded to a file | +0.101 | [+0.037, +0.180] | 60 | 269 | 829 |
| offloaded output never opened | +0.050 | [-0.076, +0.179] | 30 | 50 | 118 |
| input tokens above task median | +0.069 | [+0.025, +0.116] | 75 | 669 | 704 |
| peak prompt above task median | +0.113 | [+0.051, +0.177] | 75 | 444 | 499 |
| tool calls above task median | +0.009 | [-0.034, +0.055] | 75 | 647 | 728 |
| handbook docs above task median | +0.025 | [-0.019, +0.074] | 74 | 502 | 855 |
| re-read a handbook doc | +0.028 | [-0.028, +0.086] | 75 | 419 | 956 |

Most of these compare different agents on the same task, so stronger agents can drive the result. Truncation is set by the harness (Codex truncates long outputs in nearly every trial), so the next table compares trials within the same task and the same harness.

| signal (within task and harness) | pass-rate difference | 95% interval | strata | trials with | trials without |
|---|---|---|---|---|---|
| tokens hidden by truncation above median (Codex only) | -0.007 | [-0.073, +0.060] | 75 | 149 | 150 |
| any tool output truncated (openai-agents and hermes) | +0.186 | [+0.090, +0.308] | 26 | 53 | 51 |
| any output offloaded to a file (Claude Code) | +0.041 | [-0.060, +0.148] | 46 | 157 | 211 |
| peak prompt above median | +0.106 | [+0.057, +0.157] | 150 | 407 | 461 |
| input tokens above median | +0.071 | [+0.032, +0.112] | 223 | 582 | 608 |

In Codex trials with truncation, the median trial hid 74,237 tokens of tool output across 4 truncated outputs.


## How failed trials fail

1025 failed trials. Share of checks passed in failed trials: at least 90% in 19%, 70% to 90% in 37%, under 70% in 43%. A pass requires every check to pass.

| domain | failed trials | most common failed check groups (share of failed trials) |
|---|---|---|
| cm | 336 | judge:cm.assessment.quality (82%), judge:cm.stage_coherence (81%), judge:cm.outreach.quality (57%), cm (36%), judge:cm.care_plan.quality (34%) |
| pa_provider | 342 | new_referral (100%), provider_pre_submission (82%), cross (38%), judge:provider_pre_submission.submission.service_request (10%), judge:provider_pre_submission.submission.rationale_scope (6%) |
| pa_um | 347 | outcome (65%), judge:md_review (62%), md (52%), judge:nurse_review (46%), triage (11%) |

Caveat: these are correlations across existing runs. They point to where to look, not to causes. The controlled runs (same agent, context policy changed) are what test causes.
