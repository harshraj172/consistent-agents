from __future__ import annotations

import yaml
import json
import random
import sys
from tqdm.auto import tqdm
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from consistent_agents.data_models import (BenchmarkItem,
                                           EvalConfig, 
                                           EvalResult, 
                                           ExampleResult)
from consistent_agents.utils import (resolve_object, 
                                    maybe_instantiate)
from consistent_agents.benchmarks import BaseBenchmark
from consistent_agents.environments import BaseEnvironment

    
def load_benchmark_from_config(bm_cfg: Dict[str, Any]) -> Tuple[BaseBenchmark, List[BenchmarkItem]]:
    """Create a benchmark instance from config and return items."""
    path = bm_cfg.get("path")
    params = bm_cfg.get("params", {})
    if not path:
        raise ValueError("benchmark.path must be provided in config.yaml")
    
    obj = resolve_object(path)
    benchmark = maybe_instantiate(obj, params)
    if hasattr(benchmark, "load") and callable(getattr(benchmark, "load")):
        benchmark.load()

    items: List[BenchmarkItem] = []
    for ex_id, ex in enumerate(benchmark):
        prompt = ex["prompt"]
        if prompt is None:
            continue
        label = ex.get("label") if isinstance(ex.get("label"), (str, int)) else None
        items.append(BenchmarkItem(id=ex_id, prompt=str(prompt), label=str(label) if label is not None else None, env=ex["env"]))
    return benchmark, items

# perturbation
def _instantiate_perturbations(cfg_list: List[Dict[str, Any]]) -> List[Callable[[str], str]]:
    perts: List[Callable[[str], str]] = []
    for pcfg in cfg_list:
        path = pcfg.get("path")
        params = pcfg.get("params", {})
        if not path:
            continue
        obj = resolve_object(path)
        inst = maybe_instantiate(obj, params)
        if hasattr(inst, "apply") and callable(getattr(inst, "apply")):
            perts.append(lambda text, inst=inst: inst.apply(text))
        elif callable(inst):
            perts.append(inst)
        else:
            raise TypeError(f"Perturbation at {path} must be callable or implement .apply(text)")
    return perts


def generate_perturbations(
    text: str,
    perturb_fns: List[Tuple[Callable[[str], str]], Dict],
    n: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """Return list of (name, perturbed_text) for the given input text."""
    rng = random.Random(seed)
    if not perturb_fns:
        return []
    perturb_fns = perturb_fns*n
    results: List[Tuple[str, str]] = []
    for (fn, name) in perturb_fns:
        perturbed = fn(text)
        results.append((name, perturbed))
    return results


# agent
def resolve_agent_callable(cfg: Dict[str, Any]) -> Callable[[str], str]:
    """Resolve agent from config into a callable(prompt)"""
    path = cfg.get("path")
    params = cfg.get("params", {})
    if not path:
        raise ValueError("agent.path must be provided in config.yaml")

    obj = resolve_object(path)
    
    if isinstance(obj, type):
        model = None
        if cfg.get("model"):
            mpath = cfg["model"].get("path")
            mparams = cfg["model"].get("params", {})
            if mpath:
                model = maybe_instantiate(resolve_object(mpath), mparams)
        else:
            raise ValueError("agent.model.path must be provided in config.yaml")
        
        instance = obj(model=model, **params)

        if hasattr(instance, "run") and callable(getattr(instance, "run")):
            def _runner(prompt: str, env: BaseEnvironment) -> str:
                res = instance.run(prompt, env)
                if isinstance(res, tuple) and res:
                    return str(res[-1])
                return str(res)

            return _runner

    raise TypeError(
        "Agent must refer to a function(prompt)->str or a class with run(str)->Any"
    )


# Evaluation loop
def evaluate(
    items: List[BenchmarkItem],
    agent_fn: Callable[[str], str],
    score_fn: Callable[[str], dict],
    config: EvalConfig,
    perturb_fns: List[Callable[[str], str]],
) -> EvalResult:
    examples: List[ExampleResult] = []
    consistent_count, correct_count, total = 0, 0, 0

    for item in tqdm(items, desc="Evaluating", unit="ex"):
        base_output = agent_fn(item.prompt, item.env)
        perts = generate_perturbations(
            item.prompt,
            perturb_fns,
            n=config.n_perturbations,
            seed=config.seed,
        )
        perturbed_outputs: List[Dict[str, Any]] = []
        for p_type, p_text in perts:
            out = agent_fn(p_text, item.env)
            perturbed_outputs.append({
                "type": p_type,
                "text": p_text,
                "output": out,
            })
        result = score_fn(
            item.id, base_output, [po["output"] for po in perturbed_outputs]
        )
        consistent_count_per_row, correct_count_per_row, total_per_row = \
            result["consistent_count"], result["correct_count"], result["total"]
        correct_count += correct_count_per_row
        consistent_count += consistent_count_per_row
        total += total_per_row
        examples.append(
            ExampleResult(
                id=item.id,
                base_output=base_output,
                perturbed_outputs=perturbed_outputs,
                consistency=consistent_count_per_row/total_per_row,
                accuracy=correct_count_per_row/total_per_row,
            )
        )

    return EvalResult(
        config=asdict(config),
        consistency=consistent_count/total,
        accuracy=correct_count/total,
        total=total,
        examples=examples,
    )


def _load_config(config_path: str | Path) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: Optional[List[str]] = None) -> int:
    cfg_path = Path(argv[0]) if argv else Path("config.yaml")
    raw_cfg = _load_config(cfg_path)
    n_perturbations = int(raw_cfg.get("eval", {}).get("n_perturbations", 5))
    eval_cfg = EvalConfig(
        n_perturbations=n_perturbations,
        seed=int(raw_cfg.get("eval", {}).get("seed", 42)),
    )

    # Benchmark
    benchmark, items = load_benchmark_from_config(raw_cfg.get("benchmark", {}))
    score_fn = getattr(benchmark, "score")
    
    # Agent
    agent_fn = resolve_agent_callable(raw_cfg.get("agent", {}))

    # Perturbations
    perturb_cfgs = raw_cfg.get("perturbations", [])
    if isinstance(perturb_cfgs, dict):
        perturb_cfgs = [perturb_cfgs]
    perturb_fns = _instantiate_perturbations(perturb_cfgs)
    perturb_fns = [(fn, cfg) for fn, cfg in zip(perturb_fns, perturb_cfgs)]

    # Evaluate
    result = evaluate(items, agent_fn, score_fn, eval_cfg, perturb_fns)

    payload = {
        "config": result.config,
        "total": result.total,
        "consistency": result.consistency,
        "accuracy": result.accuracy,
        "examples": [
            {
                "id": ex.id,
                "base_output": ex.base_output,
                "perturbed_outputs": ex.perturbed_outputs,
                "consistency": ex.consistency,
                "accuracy": ex.accuracy,
            }
            for ex in result.examples
        ],
    }

    # Output
    out_path_str = raw_cfg.get("output", {}).get("path") or raw_cfg.get("out")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if out_path_str:
        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote results to {out_path}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
