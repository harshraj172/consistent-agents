# Cost Estimation Scratchpad

## 1. What we're running

**Two benchmarks with 3 perturbations per example:**

### Benchmark 1: SWE-bench Verified — Codex + GPT-5-mini (OpenAI API)
- Agent: Codex (uses OpenAI `/v1/responses` API)
- Model: `openai/gpt-5-mini`
- Perturbations: injection_swe, noise, paraphrase, translation, var_rename (5 types)
- n_perturbations: 3 per example
- N: 500 examples per perturbation type
- Trials per example: 4 (1 base + 3 perturbed)
- Total trials: 500 × 5 × 4 = 10,000
- **Status**: Paused (OpenAI quota exhausted). var_rename has 93 valid examples.

### Benchmark 2: SWE-bench Verified — OpenHands + Kimi K2 (OpenRouter)
- Agent: OpenHands (CodeActAgent)
- Model: `openrouter/moonshotai/kimi-k2` (Kimi K2 0711)
- Perturbations: injection_swe, noise, paraphrase, translation, linear_mcp (5 types)
- n_perturbations: 3 per example
- N: 500 examples per perturbation type
- Trials per example: 4 (1 base + 3 perturbed)
- Total trials: 500 × 5 × 4 = 10,000
- **Status**: Running. 46 valid examples total so far.

---

## 2. OpenRouter (Kimi K2) — Actual spend

### Source: OpenRouter API `/v1/auth/key`
- `usage_monthly`: **$80.63** (as of 2026-03-19)
- `usage_daily`: **$0.56** (today so far)

### Caveat
This $80.63 is the TOTAL spend on this API key, ever. It includes:
- The first batch of runs (Mar 16-17) where credits ran out → most trials failed with 402
- The second batch (Mar 18) where credits ran out again → more 402 failures
- The current third batch (just started Mar 19)
- Any other usage on this key (testing, etc.)

**We cannot separate Kimi K2 eval spend from other usage on this key.** The $80.63 is an upper bound on eval cost.

### Valid trials completed
| Run | Valid examples | Valid trials (×4) |
|-----|---------------|-------------------|
| injection_swe | 10 | 40 |
| noise | 8 | 32 |
| paraphrase | 9 | 36 |
| translation | 10 | 40 |
| linear_mcp | 9 | 36 |
| **Total** | **46** | **184** |

### Estimated cost per trial (from actual usage)
- Total spend: $80.63 (upper bound, includes failed 402 requests which cost $0)
- Valid trials: 184
- Upper bound per valid trial: $80.63 / 184 = **$0.438/trial**

But this overestimates because:
1. Failed 402 requests still consume some tokens before failing
2. Other non-eval usage may be included

### Better estimate: from Kimi K2 pricing + typical OpenHands usage
- Kimi K2 pricing: $0.55/M input, $2.20/M output
- OpenHands CodeActAgent: ~50 steps per trial
- Each step sends growing context (system prompt + conversation history + observation)
- Approximate token usage per trial:
  - Input: ~50 API calls × avg 30K tokens/call = ~1.5M input tokens
  - Output: ~50 API calls × avg 500 tokens/call = ~25K output tokens
  - Cost: (1.5M × $0.55 + 25K × $2.20) / 1M = $0.825 + $0.055 = **~$0.88/trial**

**NOTE**: The 30K avg input is a rough estimate. Early steps have ~5K context, later steps have ~60K+.
The actual average depends heavily on conversation length. This is uncertain.

### Cross-check
- $0.88/trial × 184 valid trials = $162 (theoretical)
- Actual spend = $80.63
- Ratio: $80.63 / $162 = 0.50

This suggests either:
1. Average input per call is lower (~15K instead of 30K), OR
2. Many trials terminate early (some examples are simple), OR
3. Significant spend went to 402-failed trials that used partial tokens

**Best estimate per valid trial: ~$0.44 - $0.88** (wide range due to uncertainty)

---

## 3. Projected cost to complete Kimi K2 runs

### Remaining work
| Run | Valid done | Remaining | Remaining trials |
|-----|-----------|-----------|-----------------|
| injection_swe | 10 | 490 | 1,960 |
| noise | 8 | 492 | 1,968 |
| paraphrase | 9 | 491 | 1,964 |
| translation | 10 | 490 | 1,960 |
| linear_mcp | 9 | 491 | 1,964 |
| **Total** | **46** | **2,454** | **9,816** |

### Cost projection
Using actual spend per trial: $80.63 / 184 = $0.438/trial (lower bound)
- Remaining: 9,816 × $0.438 = **$4,299**

Using theoretical estimate: $0.88/trial (upper bound)
- Remaining: 9,816 × $0.88 = **$8,638**

**Estimated remaining cost: $4,300 - $8,600**

**IMPORTANT**: These numbers are highly uncertain. The actual cost per trial varies enormously:
- Simple examples (agent gives up in 5 steps): ~$0.05
- Complex examples (agent uses all 50 steps with 100K context): ~$3+
- The 46 completed examples may be biased toward simpler ones (they completed first)

---

## 4. OpenAI (GPT-5-mini) cost — CANNOT ESTIMATE

### What we know
- Model: `openai/gpt-5-mini` via Codex agent
- Codex uses OpenAI `/v1/responses` API (different from chat completions)
- var_rename completed 93 valid examples out of 442 attempted (349 quota-failed)

### What we DON'T know
- OpenAI pricing for GPT-5-mini via `/v1/responses` API
- Token usage per Codex trial (Codex agent is different from OpenHands)
- Total spend on OpenAI API key (no billing API access)

**CANNOT ESTIMATE OpenAI cost.** Would need:
1. GPT-5-mini pricing (check OpenAI billing dashboard)
2. Token usage from Codex trajectories (these don't record token counts reliably)
3. Access to OpenAI usage dashboard

---

## 5. Paraphrase/Translation perturbation cost (GPT-5-nano)

Both paraphrase and translation perturbations use `gpt-5-nano` via OpenAI to generate
the perturbed prompts (back-translation / paraphrasing). This is in ADDITION to the
agent model cost.

### Per perturbation
- Each example generates 3 perturbations
- Each perturbation: 1 API call to gpt-5-nano (~1K input, ~1K output tokens)
- GPT-5-nano pricing: UNKNOWN (need to check OpenAI dashboard)

### Scale
- Kimi K2 runs: 500 examples × 2 types (paraphrase + translation) × 3 perts = 3,000 calls
- GPT-5-mini runs: same = 3,000 calls
- Total: ~6,000 GPT-5-nano API calls

**Cost likely negligible** compared to agent costs (nano is very cheap), but exact number UNKNOWN.

---

## 6. Summary

| Item | Spent | Remaining | Total |
|------|-------|-----------|-------|
| OpenRouter (Kimi K2 agent) | $80.63 | $4,300-$8,600 | $4,400-$8,700 |
| OpenAI (GPT-5-mini agent) | UNKNOWN | UNKNOWN | UNKNOWN |
| OpenAI (GPT-5-nano perturbations) | UNKNOWN | Negligible | Negligible |
| **Total** | **$80.63 + UNKNOWN** | **$4,300-$8,600 + UNKNOWN** | **UNKNOWN** |

### Key uncertainties
1. **Cost per trial varies 10-100x** depending on example complexity
2. **OpenAI costs completely unknown** — no billing API access
3. **The $80.63 includes wasted spend** on quota-failed requests
4. **46 completed examples may not be representative** of the full 500

### What would help
- Access to OpenAI usage dashboard for GPT-5-mini spend
- Run 50 more examples and re-estimate cost per trial with larger sample
- Check OpenRouter billing page for detailed per-model breakdown
