# Spider2-DBT + OpenHands + Kimi K2: Setup and Run Guide

This guide walks you through running Spider2-DBT consistency experiments with OpenHands (CodeActAgent) and Kimi K2 from scratch. It assumes you are setting up on a fresh machine.

## 1. Clone and install the repo

```bash
git clone https://github.com/harshraj172/consistent-agents.git
cd consistent-agents
pip install -e .
pip install word2word gdown
```

You also need [Harbor](https://github.com/bespokelabsai/harbor) installed:

```bash
pip install harbor-ai
```

And Docker must be running (the benchmark uses Docker containers for each task):

```bash
docker info  # should print Docker server info
```

## 2. Download Spider2-DBT data

The benchmark needs two zip files from Google Drive and the Spider2 GitHub repo.

```bash
mkdir -p .cache
cd .cache
gdown 'https://drive.google.com/uc?id=1N3f7BSWC4foj-V-1C9n8M2XmgV7FOcqL'  # DBT_start_db.zip (361 MB)
gdown 'https://drive.google.com/uc?id=1s0USV_iQLo4oe05QqAMnhGGp5jeejCzp'  # dbt_gold.zip (653 MB)
cd ..
```

The Spider2 GitHub repo (~40K files) will be auto-cloned into `.cache/Spider2/` on first run. This takes a few minutes.

## 3. Set API keys

Kimi K2 is accessed via OpenRouter:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

You can get one at https://openrouter.ai/keys. Make sure you have credits loaded -- each task costs roughly $0.50-1.00 on Kimi K2.

## 4. Create the config files

Create the directory for configs:

```bash
mkdir -p src/consistent_agents/config/openhands-kimik2-spider2-dbt
```

### 4a. Baseline config (no perturbation)

Create `src/consistent_agents/config/openhands-kimik2-spider2-dbt/baseline.yaml`:

```yaml
eval:
  n_perturbations: 0
  seed: 42

benchmark:
  path: consistent_agents.benchmarks.harbor_spider2_dbt:HarborSpider2DBTBenchmark
  params:
    split: "all"
    cache_dir: .cache
    tasks_root: .harbor_tasks/spider2_dbt_openhands_kimik2_baseline
    force_reclone: false
    overwrite: false
    include_db_variants: false
    db_perturb_seed: 42

perturbations: []

harbor:
  n_concurrent: 4
  task:
    template_path: .harbor_tasks/spider2_dbt_openhands_kimik2_baseline
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
  path: outputs/harbor-spider2-dbt-openhands-kimik2-baseline
```

### 4b. Perturbation configs

Create one config per perturbation. The only differences from baseline are:
- `include_db_variants: true` (generates both base and perturbed task dirs)
- `db_perturb_spec` with the perturbation type
- Unique `tasks_root` and `output.path`

**timestamp_format** -- `src/consistent_agents/config/openhands-kimik2-spider2-dbt/timestamp.yaml`:

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

**header_shuffle** -- same as above but change:
- `tasks_root: .harbor_tasks/spider2_dbt_openhands_kimik2_header_shuffle`
- `template_path: .harbor_tasks/spider2_dbt_openhands_kimik2_header_shuffle`
- `db_perturb_spec.name: header_shuffle`
- `perturbation_types: [header_shuffle]`
- `output.path: outputs/harbor-spider2-dbt-openhands-kimik2-header-shuffle`

**header_translate** -- same pattern:
- `tasks_root: .harbor_tasks/spider2_dbt_openhands_kimik2_header_translate`
- `template_path: .harbor_tasks/spider2_dbt_openhands_kimik2_header_translate`
- `db_perturb_spec.name: header_translate`
- `perturbation_types: [header_translate]`
- `output.path: outputs/harbor-spider2-dbt-openhands-kimik2-header-translate`

## 5. Run the experiments

### 5a. Validate with Oracle first (optional but recommended)

The oracle agent uses ground-truth answers. If perturbations are working correctly, oracle should get ~97% on perturbed tasks:

```bash
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-oracle-timestamp.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-oracle-header-shuffle.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/harbor-spider2-dbt-oracle-header-translate.yaml
```

Each takes ~2 hours (128 items, `n_concurrent=1`).

### 5b. Run Kimi K2

```bash
# Baseline
python -m consistent_agents.eval_harbor \
  src/consistent_agents/config/openhands-kimik2-spider2-dbt/baseline.yaml

# Perturbations (can run in parallel on separate terminals)
python -m consistent_agents.eval_harbor \
  src/consistent_agents/config/openhands-kimik2-spider2-dbt/timestamp.yaml

python -m consistent_agents.eval_harbor \
  src/consistent_agents/config/openhands-kimik2-spider2-dbt/header-shuffle.yaml

python -m consistent_agents.eval_harbor \
  src/consistent_agents/config/openhands-kimik2-spider2-dbt/header-translate.yaml
```

Each perturbation run evaluates 128 items (64 base + 64 perturbed). With `n_concurrent=4` expect ~3-4 hours per run.

To run in background so it survives terminal disconnects:

```bash
nohup python -m consistent_agents.eval_harbor \
  src/consistent_agents/config/openhands-kimik2-spider2-dbt/timestamp.yaml \
  > /tmp/spider2-kimik2-timestamp.log 2>&1 &
```

### 5c. Monitor progress

Check incremental results:

```bash
python3 -c "
import json, glob
for name in ['baseline', 'timestamp', 'header-shuffle', 'header-translate']:
    files = sorted(glob.glob(f'outputs/harbor-spider2-dbt-openhands-kimik2-{name}/*/result.json'))
    if files:
        d = json.load(open(files[-1]))
        n = d.get('completed_examples', len(d.get('examples', [])))
        total = d.get('total', '?')
        acc = round(d.get('accuracy', 0), 3)
        status = d.get('status', 'unknown')
        print(f'{name}: {n}/{total} done, accuracy={acc}, status={status}')
    else:
        print(f'{name}: no results yet')
"
```

### 5d. Resume interrupted runs

If a run crashes or you need to restart the machine:

```bash
python -m consistent_agents.eval_harbor \
  src/consistent_agents/config/openhands-kimik2-spider2-dbt/timestamp.yaml \
  --resume outputs/harbor-spider2-dbt-openhands-kimik2-timestamp/<timestamp_dir>
```

This skips already-completed tasks and continues from where it left off.

## 6. Understanding the output

Each run produces:

```
outputs/harbor-spider2-dbt-openhands-kimik2-<perturbation>/<timestamp>/
  result.json              # Scores and per-example results
  trajectory.json          # Index pointing to individual trajectory files
  trajectories/            # One file per task
    <task_id>__base.json
    <task_id>__perturbation.json
```

### result.json structure

```json
{
  "status": "completed",
  "total": 128,
  "accuracy": 0.234,
  "consistency": 0.812,
  "examples": [
    {
      "id": "google_play001",
      "base_output": "1.0",
      "is_perturbed": false,
      "accuracy": 1.0,
      "consistency": 1.0
    },
    ...
  ]
}
```

- `accuracy` = fraction of tasks where the agent's output scored > 0 (via Harbor verifier)
- `consistency` = fraction of tasks where base and perturbed runs agreed (both pass or both fail)
- `is_perturbed` = whether this example ran on a perturbed database

### Extracting base vs perturbed accuracy

```python
import json

d = json.load(open("outputs/.../result.json"))
base = [e for e in d["examples"] if not e.get("is_perturbed")]
pert = [e for e in d["examples"] if e.get("is_perturbed")]

base_acc = sum(e["accuracy"] for e in base) / len(base)
pert_acc = sum(e["accuracy"] for e in pert) / len(pert)
print(f"Base accuracy: {base_acc:.3f} ({len(base)} tasks)")
print(f"Perturbed accuracy: {pert_acc:.3f} ({len(pert)} tasks)")
```

## 7. How the perturbations work

Spider2-DBT perturbations modify the DuckDB database before the agent sees it:

| Perturbation | What it does | Example |
|---|---|---|
| **timestamp_format** | Changes timestamp column output formats via `strftime()` | `2024-01-15 10:30:00` -> `15/01/2024 10:30:00` |
| **header_shuffle** | Shuffles tokens in multi-word column names | `order_date` -> `date_order` |
| **header_translate** | Translates column name tokens to FR/ZH/JA/ES | `order_date` -> `commande_date` |

The agent receives the same instruction (SQL/dbt task) but the underlying database schema is different. A consistent agent should produce the same pass/fail result regardless of superficial schema changes.

## 8. Key config parameters

| Parameter | What it controls |
|---|---|
| `include_db_variants` | `true` = run both base + perturbed tasks; `false` = base only |
| `perturbation_types` | List of which perturbations to apply. Omit to apply all 3 |
| `tasks_root` | Where task directories are generated. **Must be unique per experiment** |
| `template_path` | Must match `tasks_root` |
| `n_concurrent` | Number of parallel Docker containers. 4 is safe; increase if you have more CPU/RAM |
| `rewrite_instruction` | `true` for OpenHands (it needs reformatted instructions); `false` for Codex/Oracle |
| `db_perturb_seed` | Random seed for perturbation reproducibility |

## 9. Reference: existing results

See `final_results.md` for all completed and in-progress results across benchmarks.

### Spider2-DBT scores so far (no Kimi K2 yet)

| Agent | Baseline | timestamp | header_shuffle | header_translate |
|---|---|---|---|---|
| **Oracle** | 1.000 | 0.969 | 0.953 | 0.953 |
| **Codex + GPT-5-mini** | 0.250 | 0.203 | 0.125 | 0.141 |
| **OpenHands + Kimi K2** | -- | -- | -- | -- |

## 10. Troubleshooting

### `FileNotFoundError: Required zip not found: .cache/DBT_start_db.zip`
You forgot to download the zip files. See step 2.

### `ModuleNotFoundError: No module named 'word2word'`
Run `pip install word2word`.

### Tasks fail with Docker errors
Make sure Docker is running (`docker info`) and you have enough disk space (~50GB free recommended for all task containers).

### Agent exits with NonZeroAgentExitCodeError
This usually means the agent timed out or crashed inside the container. Check the trajectory file for that task in `trajectories/<task_id>__base.json` for details.

### Run seems stuck at 0%
The first run takes extra time to clone the Spider2 repo (~40K files) and generate task directories. The progress bar will start moving after setup completes.
