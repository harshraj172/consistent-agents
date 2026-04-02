# Consistency Evaluation Results

Consistency = per-example binary match rate: 1.0 if base and perturbed runs agree (both pass or both fail), 0.0 otherwise. Averaged across all examples.

**Note:** All results below are with **1 perturbed run** per example (n_perturbations=1).

## SWE-bench Verified

| Agent | Perturbation | N | Base Acc | Pert Acc | Consistency |
|-------|-------------|---|----------|----------|-------------|
| GPT5mini | linear-mcp | 500 | 0.660 | 0.456 | 0.516 |
| GPT5mini | injection_swe | 500 | 0.472 | 0.486 | 0.814 |

## Spider2-DBT

| Agent | Perturbation | N | Base Acc | Pert Acc | Consistency |
|-------|-------------|---|----------|----------|-------------|
| GPT5mini | header-shuffle | 64 | 0.141 | 0.109 | 0.969 |
| GPT5mini | header-translate | 64 | 0.141 | 0.109 | 0.969 |
| GPT5mini | timestamp | 64 | 0.172 | 0.188 | 0.828 |

## BFCL

| Agent | Perturbation | N | Base Acc | Pert Acc | Consistency |
|-------|-------------|---|----------|----------|-------------|
| Codex | perturbs | 12 | 0.833 | 0.667 | 0.750 |

## SWE-bench Verified — GPT5mini 3 perturbations (in progress)

Consistency: per-example, if base and **all 3** perturbed runs agree → 1, else 0.

| Agent | Perturbation | N | Base Acc | Mean Pert Acc | Consistency | Status |
|-------|-------------|---|----------|---------------|-------------|--------|
| Codex+GPT5mini | injection_swe | 0/500 | — | — | — | Cleared, ready to restart |
| Codex+GPT5mini | noise | 0/500 | — | — | — | Cleared, ready to restart |
| Codex+GPT5mini | paraphrase | 0/500 | — | — | — | Cleared, ready to restart |
| Codex+GPT5mini | translation | 0/500 | — | — | — | Cleared, ready to restart |
| Codex+GPT5mini | var_rename | 93/500 | 0.677 | 0.480 | 0.409 | Filtered (349 quota-failed examples removed), ready to resume |

**Data cleanup (2026-03-14)**: Removed 349/442 var_rename examples where agent had 0 messages due to OpenAI quota exhaustion. Remaining 93 examples have valid agent execution (base_acc=0.677, consistent with 1-pert runs). All 4 prompt-level runs cleared — their results were entirely from quota-degraded period. All configs set to `n_concurrent: 1`. Harbor-trials cleanup cron running every 30 min.

## SWE-bench Verified — OpenHands + Kimi K2 (0711) 3 perturbations (in progress)

Agent: OpenHands (CodeActAgent) | Model: moonshotai/kimi-k2 (0711) via OpenRouter | n_perturbations: 3

Consistency: per-example, if base and **all 3** perturbed runs agree → 1, else 0.

| Agent | Perturbation | N (new) | N (total valid) | Base Acc | Mean Pert Acc | Consistency | Status |
|-------|-------------|---------|----------------|----------|---------------|-------------|--------|
| OpenHands+KimiK2 | injection_swe | 56/381 | 56+119=175 | 0.589 | 0.601 | 0.732 | Running |
| OpenHands+KimiK2 | noise | 67/393 | 67+107=174 | 0.672 | 0.617 | 0.791 | Running |
| OpenHands+KimiK2 | paraphrase | 68/476 | 68 | 0.471 | 0.505 | 0.779 | Running |
| OpenHands+KimiK2 | translation | 71/475 | 71 | 0.606 | 0.615 | 0.775 | Running |
| OpenHands+KimiK2 | linear_mcp | 52/500 | 52 | 0.654 | 0.090 | 0.327 | Running |

**N (new)**: Examples completed in current (valid) run. **N (total valid)**: Includes retained valid examples from earlier runs (injection_swe: 119, noise: 107). Paraphrase/translation restarted fresh due to broken perturbation prompts. Linear_mcp restarted fresh due to MCP registration fix.

**OpenRouter spend**: $1,186.67 total as of 2026-03-26. Burn rate ~$145/day with 5 parallel runs.

**Key observations**:
- **injection_swe**: Most robust — pert_acc (60.1%) ≈ base_acc (58.9%), consistency 73.2%
- **noise**: Perturbation drops accuracy moderately (61.7% vs 67.2%), high consistency 79.1%
- **paraphrase**: High consistency (77.9%), pert_acc slightly above base (50.5% vs 47.1%)
- **translation**: High consistency (77.5%), pert_acc close to base (61.5% vs 60.6%)
- **linear_mcp**: Very low pert accuracy (9.0%) — agent can't retrieve issues via MCP, consistency 32.7%

**Bugs fixed during runs**:
- `rewrite_instruction: false` → `true` for prompt-level perturbations (injection_swe, noise, paraphrase, translation)
- gpt-5-nano `temperature=0.7` unsupported → added `temperature: 1.0` + `reasoning_effort: minimal`
- Paraphrase/translation prompt templates rewritten for long-form text (was designed for single sentences)
- `max_tokens: 500` → `4096` for perturbation models
- OpenHands `--prerelease=allow` fix for openhands-ai dependency resolution
- linear_mcp `mcp_servers` written to wrong TOML path (top-level vs `[environment]`)

## Notes

- GPT5mini Spider2-DBT consistency is high (0.97) despite low accuracy because the agent fails consistently on both base and perturbed inputs.
- Together AI abandoned: Codex uses `/v1/responses` API (incompatible), OpenHands+Qwen2.5-7B too small for SWE-bench tool use, Kimi K2.5/DeepSeek V3.1 had severe API timeouts.
- **Bug fix (2026-03-12)**: Previous 3-pert runs showed 0% accuracy due to `uv: command not found` in test.sh verifier. The Dockerfile installed `uv` but didn't add it to PATH. Fixed by adding `ENV PATH="/root/.local/bin:${PATH}"` to all 500 task Dockerfiles. Restarting all runs.
