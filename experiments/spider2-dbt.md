# Spider2-DBT Consistency Experiments

## Overview

This document describes how to run Spider2-DBT consistency experiments with isolated perturbations. The benchmark evaluates text-to-SQL agents on real-world dbt projects, and we test robustness via 3 database-level perturbations:

| Perturbation | What it does |
|---|---|
| **timestamp_format** | Changes timestamp column output formats (e.g. `%d/%m/%Y %H:%M:%S`) |
| **header_shuffle** | Shuffles token order in column names (e.g. `order_date` -> `date_order`) |
| **header_translate** | Translates column name tokens to other languages (fr/zh/ja/es) |

Each perturbation modifies the DuckDB database in-place before the agent runs.

## Prerequisites

### 1. Install dependencies

```bash
pip install -e .
pip install word2word multipart
```

### 2. Download Spider2-DBT data

Two zip files are required from Google Drive:

```bash
pip install gdown
mkdir -p .cache
cd .cache
gdown 'https://drive.google.com/uc?id=1N3f7BSWC4foj-V-1C9n8M2XmgV7FOcqL'  # DBT_start_db.zip
gdown 'https://drive.google.com/uc?id=1s0USV_iQLo4oe05QqAMnhGGp5jeejCzp'  # dbt_gold.zip
cd ..
```

The Spider2 GitHub repo will be auto-cloned into `.cache/Spider2` on first run.

### 3. Set API keys

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Running Experiments

### Step 1: Validate perturbations with Oracle agent

The Oracle agent uses ground-truth answers. Run it first to confirm perturbations work correctly (expected: ~96-98% accuracy on perturbed tasks).

```bash
# Run each perturbation individually with oracle
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-oracle-timestamp.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-oracle-header-shuffle.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-oracle-header-translate.yaml
```

Each run evaluates 128 items (64 base + 64 perturbed) and takes ~2 hours at `n_concurrent=1`.

**Expected oracle results:**

| Perturbation | Base Acc | Perturbed Acc | Overall |
|---|---|---|---|
| timestamp_format | 1.0 | 0.969 | 0.984 |
| header_shuffle | 1.0 | 0.953 | 0.977 |
| header_translate | 1.0 | 0.953 | 0.977 |

### Step 2: Run with Codex + GPT-5-mini

```bash
# Baseline (no perturbations)
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-codex-gpt5mini-baseline.yaml

# Each perturbation
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-codex-gpt5mini-timestamp.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-codex-gpt5mini-header-shuffle.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-codex-gpt5mini-header-translate.yaml
```

**Results (Codex + GPT-5-mini):**

| Perturbation | Base Acc | Perturbed Acc | Overall |
|---|---|---|---|
| Baseline | 29.7% | N/A | 29.7% |
| timestamp_format | 17.2% | 18.8% | 18.0% |
| header_shuffle | 14.1% | 10.9% | 12.5% |
| header_translate | 14.1% | 10.9% | 12.5% |

### Step 3: Run with OpenHands + Kimi K2

Create configs for each perturbation using OpenHands agent with Kimi K2 via OpenRouter:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"

# Baseline
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-spider2-dbt/baseline.yaml

# Each perturbation
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-spider2-dbt/timestamp.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-spider2-dbt/header-shuffle.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-spider2-dbt/header-translate.yaml
```

Example config (`openhands-kimik2-spider2-dbt/timestamp.yaml`):

```yaml
eval:
  n_perturbations: 0
  seed: 42

benchmark:
  path: consistent_agents.benchmarks.harbor_spider2_dbt:HarborSpider2DBTBenchmark
  params:
    split: "all"
    cache_dir: .cache
    tasks_root: .harbor_tasks/spider2_dbt_openhands_kimik2_timestamp
    force_reclone: false
    overwrite: false
    include_db_variants: true
    db_perturb_seed: 42
    db_perturb_spec:
      name: timestamp_format
      perturbation_types:
        - timestamp_format

perturbations: []

harbor:
  n_concurrent: 4
  task:
    template_path: .harbor_tasks/spider2_dbt_openhands_kimik2_timestamp
    instruction_file: instruction.md
    workdir: .harbor_runs
    keep: false
    rewrite_instruction: true
  agent:
    name: openhands
    model_name: openrouter/moonshotai/kimi-k2
  environment:
    type: docker
    delete: true
  verifier: {}
  trials_dir: harbor-trials
  timeout_multiplier: 1.0
  reward_key: reward

output:
  path: outputs/harbor-spider2-dbt-openhands-kimik2-timestamp
```

### Step 4: Run with a different agent/model

To run with your own agent and model, create a new config based on any existing one:

```yaml
eval:
  n_perturbations: 0
  seed: 42

benchmark:
  path: consistent_agents.benchmarks.harbor_spider2_dbt:HarborSpider2DBTBenchmark
  params:
    split: "all"
    cache_dir: .cache
    tasks_root: .harbor_tasks/spider2_dbt_<your_experiment_name>
    force_reclone: false
    overwrite: false
    include_db_variants: true          # true = run both base + perturbed
    db_perturb_seed: 42
    db_perturb_spec:
      name: <perturbation_name>        # e.g. timestamp_format
      perturbation_types:              # pick one or more:
        - timestamp_format
        # - header_shuffle
        # - header_translate

perturbations: []

harbor:
  n_concurrent: 4
  task:
    template_path: .harbor_tasks/spider2_dbt_<your_experiment_name>
    instruction_file: instruction.md
    workdir: .harbor_runs
    keep: false
    rewrite_instruction: false
  agent:
    name: <agent_name>                 # e.g. codex, openhands
    model_name: <model_name>           # e.g. openai/gpt-5-mini
  environment:
    type: docker
    delete: true
  verifier: {}
  trials_dir: harbor-trials
  timeout_multiplier: 1.0
  reward_key: reward

output:
  path: outputs/<your_output_dir>
```

Key config options:
- **`include_db_variants: true`** — generates both base and perturbed task dirs
- **`perturbation_types`** — list of which perturbations to apply (omit to apply all 3)
- **`tasks_root`** — use a unique path per experiment to avoid task dir collisions
- **`n_concurrent`** — number of parallel evaluations

## Resuming interrupted runs

If a run crashes or is interrupted, resume from where it left off:

```bash
python -m consistent_agents.eval_harbor <config.yaml> --resume <output_dir>/<timestamp_dir>
```

Example:
```bash
python -m consistent_agents.eval_harbor \
  src/consistent_agents/config/harbor-spider2-dbt-codex-gpt5mini-timestamp.yaml \
  --resume outputs/harbor-spider2-dbt-codex-gpt5mini-timestamp/harbor-spider2-dbt-codex-gpt5mini-timestamp-2026-03-06::05-00-37
```

This skips already-completed task IDs and continues with remaining tasks. You can also change `n_perturbations` or `n_concurrent` in the config before resuming.

## Output structure

```
outputs/<run_name>/<timestamp>/
  result.json              # Scores + per-example results
  trajectory.json          # Index file pointing to individual trajectories
  trajectories/            # Per-task trajectory files
    <task_id>__base.json
    <task_id>__perturbation.json
```

## How perturbations work

The `duckdb_perturb.py` module modifies the DuckDB database before the agent sees it:

1. **timestamp_format**: Wraps timestamp columns with `strftime()` using an alternate format
2. **header_shuffle**: Splits multi-token column names on `_` and shuffles the tokens
3. **header_translate**: Translates column name tokens to French/Chinese/Japanese/Spanish using word2word

The `perturbation_types` key in `db_perturb_spec` controls which are applied. If omitted, all 3 are applied together.
