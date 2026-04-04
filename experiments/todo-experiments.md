# Run Spider2-DBT + OpenHands + Kimi K2

## Setup

```bash
git clone https://github.com/harshraj172/consistent-agents.git
cd consistent-agents
pip install -e .

git clone https://github.com/harbor-framework/harbor.git ../harbor
pip install -e ../harbor

pip install word2word gdown

mkdir -p .cache && cd .cache
gdown 'https://drive.google.com/uc?id=1N3f7BSWC4foj-V-1C9n8M2XmgV7FOcqL'  # DBT_start_db.zip
gdown 'https://drive.google.com/uc?id=1s0USV_iQLo4oe05QqAMnhGGp5jeejCzp'  # dbt_gold.zip
cd ..

export OPENROUTER_API_KEY="your-openrouter-key"
```

## Run

```bash
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-spider2-dbt/timestamp.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-spider2-dbt/header-shuffle.yaml
python -m consistent_agents.eval_harbor src/consistent_agents/config/openhands-kimik2-spider2-dbt/header-translate.yaml
```

Each run evaluates 128 items (64 base + 64 perturbed). Takes ~3-4 hours per run.

## Resume interrupted runs

```bash
python -m consistent_agents.eval_harbor <config.yaml> --resume <output_dir>/<timestamp_dir>
```
