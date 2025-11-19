from __future__ import annotations

import json
import random
import sys
import uuid
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from tqdm.auto import tqdm

from consistent_agents.data_models import (
    AgentRunResult,
    AgentTrajectory,
    BenchmarkItem,
    EvalConfig,
    EvalResult,
    ExampleResult,
)
from consistent_agents.utils import maybe_instantiate, resolve_object
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
    perturb_fns: List[Tuple[Callable[[str], str], Dict[str, Any]]],
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
def resolve_agent_callable(cfg: Dict[str, Any]) -> Callable[[str, BaseEnvironment], AgentRunResult]:
    """Resolve agent from config into a callable(prompt, env) -> AgentRunResult."""
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
            def _runner(prompt: str, env: BaseEnvironment) -> AgentRunResult:
                res = instance.run(prompt, env)
                if isinstance(res, tuple) and res:
                    status = str(res[0])
                    output_text = str(res[-1])
                else:
                    status = "OK"
                    output_text = str(res)

                messages = deepcopy(instance.messages)
                try:
                    agent_config = asdict(instance.config)
                except TypeError:
                    agent_config = getattr(instance, "config", {})

                metadata = {
                    "agent_name": instance.__class__.__name__,
                    "agent_config": agent_config,
                    "model": instance.model.get_template_vars() if hasattr(instance.model, "get_template_vars") else {},
                    "steps_taken": sum(1 for message in messages if message.get("role") == "assistant"),
                }

                return AgentRunResult(
                    output=output_text,
                    status=status,
                    messages=messages,
                    metadata=metadata,
                )

            return _runner

    raise TypeError(
        "Agent must refer to a function(prompt)->str or a class with run(str)->Any"
    )


# Evaluation loop
def evaluate(
    items: List[BenchmarkItem],
    agent_fn: Callable[[str, BaseEnvironment], AgentRunResult],
    benchmark: BaseBenchmark,
    config: EvalConfig,
    perturb_fns: List[Tuple[Callable[[str], str], Dict[str, Any]]],
) -> EvalResult:
    examples: List[ExampleResult] = []
    consistent_count, correct_count, total = 0, 0, 0
    trajectories: List[AgentTrajectory] = []

    for item in tqdm(items, desc="Evaluating", unit="ex"):
        base_result = agent_fn(item.prompt, item.env)
        if isinstance(base_result, AgentRunResult):
            base_output = base_result.output
            trajectories.append(
                AgentTrajectory(
                    example_id=item.id,
                    variant="base",
                    prompt=item.prompt,
                    output=base_output,
                    status=base_result.status,
                    messages=base_result.messages,
                    metadata=dict(base_result.metadata),
                )
            )
        else:
            base_output = str(base_result)
        perts = generate_perturbations(
            item.prompt,
            perturb_fns,
            n=config.n_perturbations,
            seed=config.seed,
        )
        perturbed_outputs: List[Dict[str, Any]] = []
        for p_type, p_text in perts:
            pert_result = agent_fn(p_text, item.env)
            if isinstance(pert_result, AgentRunResult):
                out = pert_result.output
                pert_metadata = dict(pert_result.metadata)
                pert_metadata.setdefault("perturbation", p_type)
                trajectories.append(
                    AgentTrajectory(
                        example_id=item.id,
                        variant="perturbation",
                        prompt=p_text,
                        output=out,
                        status=pert_result.status,
                        messages=pert_result.messages,
                        metadata=pert_metadata,
                    )
                )
            else:
                out = str(pert_result)
            perturbed_outputs.append({
                "type": p_type,
                "prompt": p_text,
                "output": out,
            })
        benchmark.score(
            item.id, base_output, [po["output"] for po in perturbed_outputs]
        )
        item_score = benchmark.item_score()
        examples.append(
            ExampleResult(
                id=item.id,
                base_prompt=item.prompt,
                base_output=base_output,
                perturbed_outputs=perturbed_outputs,
                **item_score,
            )
        )

    total_score = benchmark.total_score()
    return EvalResult(
        config=asdict(config),
        **total_score,
        examples=examples,
        trajectories=trajectories,
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
    
    # Agent
    agent_fn = resolve_agent_callable(raw_cfg.get("agent", {}))

    # Perturbations
    perturb_cfgs = raw_cfg.get("perturbations", [])
    if isinstance(perturb_cfgs, dict):
        perturb_cfgs = [perturb_cfgs]
    perturb_fns = _instantiate_perturbations(perturb_cfgs)
    perturb_fns = [(fn, cfg) for fn, cfg in zip(perturb_fns, perturb_cfgs)]

    # Evaluate
    result = evaluate(items, agent_fn, benchmark, eval_cfg, perturb_fns)

    payload = {
        "config": result.config,
        "total": result.total,
        "consistency": result.consistency,
        "accuracy": result.accuracy,
        "examples": [
            {
                "id": ex.id,
                "base_prompt": ex.base_prompt,
                "base_output": ex.base_output,
                "perturbed_outputs": ex.perturbed_outputs,
                "consistency": ex.consistency,
                "accuracy": ex.accuracy,
            }
            for ex in result.examples
        ],
    }

    out_path_str = raw_cfg.get("output", {}).get("path") or raw_cfg.get("out")

    if out_path_str:
        base_out = Path(out_path_str)
        if base_out.suffix:
            base_dir = base_out.parent
            base_name = base_out.stem
        else:
            base_dir = base_out
            base_name = base_out.name

        if not base_name:
            base_name = "results"

        timestamp = datetime.now()
        run_dir_name = f"{base_name}-{timestamp.strftime('%Y-%m-%d::%H-%M-%S')}"
        run_dir = base_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        trajectory_filename = f"trajectory.json"
        trajectory_path = run_dir / trajectory_filename
        result_path = run_dir / "result.json"

        payload["result_trajectory"] = trajectory_filename

        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        trajectory_payload = {
            "created_at": timestamp.isoformat(),
            "results_dir": run_dir_name,
            "trajectory_file": trajectory_filename,
            "trajectory_count": len(result.trajectories),
            "entries": [asdict(entry) for entry in result.trajectories],
        }
        trajectory_path.write_text(
            json.dumps(trajectory_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Wrote results to {result_path} and trajectory log to {trajectory_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
