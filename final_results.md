# Consistency Evaluation Results

Last updated: 2026-04-04

Consistency = per-example binary match rate: 1.0 if base and ALL perturbed runs agree (both pass or both fail), 0.0 otherwise. Averaged across all valid examples.

## SWE-bench Verified -- Codex + GPT-5-mini

### 1-perturbation runs (complete, N=500)

| Perturbation | N | Base Acc | Pert Acc | Consistency |
|---|---|---|---|---|
| linear_mcp | 500 | 0.660 | 0.456 | 0.516 |
| injection_swe | 500 | 0.472 | 0.486 | 0.814 |

### 3-perturbation runs (stalled -- need --resume)

These runs completed 500 examples but ~370 errored with NonZeroAgentExitCodeError after a Harbor codex.py update on 2026-04-02. Only ~130 have valid results.

| Perturbation | N valid/500 | Base Acc | Pert Acc | Consistency | Status |
|---|---|---|---|---|---|
| noise | ~129/500 | 0.676 | 0.623 | 0.690 | Stalled |
| paraphrase | ~130/500 | 0.697 | 0.684 | 0.737 | Stalled |
| translation | ~112/500 | 0.705 | 0.702 | 0.741 | Stalled |
| var_rename | ~93/500 | 0.677 | 0.480 | 0.409 | Stalled |

## SWE-bench Verified -- OpenHands + Kimi K2 (0711)

### 1-perturbation runs (actively running, ~35-50% done)

Model: `openrouter/moonshotai/kimi-k2` via OpenRouter

| Perturbation | N valid/total | Base Acc | Pert Acc | Consistency | Status |
|---|---|---|---|---|---|
| injection_swe | 174/348 | 0.141 | 0.142 | 0.897 | Running |
| linear_mcp | 141/282 | 0.143 | 0.195 | 0.894 | Running |
| noise | 149/298 | 0.156 | 0.131 | 0.893 | Running |
| paraphrase | 180/360 | 0.102 | 0.179 | 0.894 | Running |
| translation | 156/312 | 0.181 | 0.148 | 0.853 | Running |
| var_rename | 160/320 | 0.193 | 0.104 | 0.812 | Running |

Note: KimiK2 base accuracy is very low (~10-19%). High consistency is largely because the agent fails consistently on both base and perturbed.

## Spider2-DBT

### Oracle (1-perturbation, N=64) -- COMPLETED

| Perturbation | Base Acc | Pert Acc | Consistency |
|---|---|---|---|
| timestamp | 1.000 | 0.969 | 0.969 |
| header_shuffle | 1.000 | 0.953 | 0.953 |
| header_translate | 1.000 | 0.953 | 0.953 |

### Codex + GPT-5-mini (N=64) -- COMPLETED

April 4 re-run:

| Perturbation | N | Base Acc | Pert Acc | Consistency |
|---|---|---|---|---|
| baseline (no pert) | 64 | 0.250 | -- | -- |
| timestamp | 64+64 | 0.266 | 0.203 | 0.844 |
| header_shuffle | 64+64 | 0.266 | 0.125 | 0.797 |
| header_translate | 64+64 | 0.266 | 0.141 | 0.812 |

### OpenHands + Kimi K2 -- NOT RUN

No configs or results exist. See `experiments/todo-experiments.md` for how to run.

## BFCL

| Agent + Model | Perturbation | N | Base Acc | Pert Acc | Consistency |
|---|---|---|---|---|---|
| Codex + GPT5mini | perturbs | 12 | 0.833 | 0.667 | 0.750 |

## Data Cleanup Log

- **2026-04-02**: Cleaned FileNotFoundError failures caused by Harbor codex.py update removing install-codex.sh.j2 template. GPT5mini: removed 355-388 failures per run. KimiK2: removed 140-466 failures per run. All runs restarted with updated Harbor.
- **2026-03-24**: Cleaned openhands-ai dependency resolution failures (--prerelease=allow fix).
- **2026-03-14**: Cleaned OpenAI quota exhaustion failures from var_rename (349/442 removed).
- **2026-03-13**: Fixed rewrite_instruction=false bug, paraphrase/translation prompt templates, gpt-5-nano temperature/reasoning_effort.
- **2026-03-12**: Fixed uv PATH bug in SWE-bench Dockerfiles.
