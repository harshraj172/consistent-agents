# consistent-agents

Evaluate the consistency of AI agents under input perturbations.

## Quick start

```bash
pip install -e .
```

## Running evaluations

All evaluations use Harbor for environment management:

```bash
python -m consistent_agents.eval_harbor <config.yaml>
```

Config files are in `src/consistent_agents/config/`. See below for specific benchmarks.

## Benchmarks

### SWE-bench Verified

```bash
# Codex + GPT-5-mini with injection_swe perturbation
python -m consistent_agents.eval_harbor src/consistent_agents/config/gpt5mini-swebench/injection_swe.yaml

# OpenHands + Kimi K2 with noise perturbation
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-swebench-3pert/noise.yaml
```

### Spider2-DBT

Requires additional setup (downloading data). See [experiments/spider2-dbt.md](experiments/spider2-dbt.md) for full instructions.

```bash
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-codex-gpt5mini-timestamp.yaml
```

### Running new experiments

To run Spider2-DBT with OpenHands + Kimi K2 from scratch, see [experiments/todo-experiments.md](experiments/todo-experiments.md).

## Results

See [final_results.md](final_results.md) for all completed and in-progress results.

## Resuming interrupted runs

```bash
python -m consistent_agents.eval_harbor <config.yaml> --resume <output_dir>/<timestamp_dir>
```
