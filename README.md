# consistent-agents
Attempt to evaluate the consistency of AI Agents.

## Test
### TruthfulQA
Test run with:
```bash
pip install -e .
python3 -m src.consistent_agents.eval src/consistent_agents/config/truthfulqa.yaml
```

### SWEBench
Test run with:
```bash
pip install -e .
python3 -m src.consistent_agents.eval src/consistent_agents/config/swebench.yaml
```

Using harbor:
```bash
python -m src.consistent_agents.eval_harbor src/consistent_agents/config/harbor-swebench-gemini.yaml 

python -m src.consistent_agents.eval_harbor src/consistent_agents/config/harbor-swebench-codex.yaml
``` 
