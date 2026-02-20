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
    item_id = ex.get("instance_id", ex_id)
    metadata = dict(ex.get("metadata") or {})

    items.append(
        BenchmarkItem(
            id=str(item_id),
            prompt=str(prompt),
            label=str(label) if label is not None else None,
            env=ex["env"],
            metadata=metadata,
            base_commit=ex.get("base_commit"),
        )
    )

    return benchmark, items

# perturbation
def _instantiate_perturbations(cfg_list: List[Dict[str, Any]]) -> List[Tuple[Any, Dict[str, Any]]]:
    perts: List[Tuple[Any, Dict[str, Any]]] = []
    for pcfg in cfg_list:
        path = pcfg.get("path")
        params = pcfg.get("params", {})
        if not path:
            continue
        obj = resolve_object(path)
        inst = maybe_instantiate(obj, params)
        if not (hasattr(inst, "apply") and callable(getattr(inst, "apply"))) and not callable(inst):
            raise TypeError(f"Perturbation at {path} must be callable or implement .apply(text)")
        perts.append((inst, pcfg))
    return perts


def generate_perturbations(
    text: str,
    perturb_instances: List[Tuple[Any, Dict[str, Any]]],
    n: int,
    seed: int,
) -> List[Tuple[str, str, Any]]:
    """Return list of (name, perturbed_text) for the given input text."""
    rng = random.Random(seed)
    if not perturb_instances:
        return []
    perturb_instances = perturb_instances * n
    results: List[Tuple[str, str, Any]] = []

    for (inst, cfg) in perturb_instances:
        name = cfg.get("name", inst.__class__.__name__)
        if hasattr(inst, "apply") and callable(getattr(inst, "apply")):
            perturbed = inst.apply(text)
        else:
            perturbed = inst(text)
        results.append((name, perturbed, inst))
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


def _save_incremental_results(
    run_dir: Path,
    config: Dict[str, Any],
    examples: List[ExampleResult],
    trajectories: List[AgentTrajectory],
    benchmark: BaseBenchmark,
    timestamp: datetime,
) -> None:
    """Save current results incrementally to disk."""
    total_score = benchmark.total_score()

    payload = {
        "config": config,
        "status": "in_progress",
        "completed_examples": len(examples),
        **{k: v for k, v in {
            "total": total_score.get("total"),
            "consistency": total_score.get("consistency"),
            "accuracy": total_score.get("accuracy"),
        }.items() if v is not None},
        "examples": [
            {
                "id": ex.id,
                "base_prompt": ex.base_prompt,
                "base_output": ex.base_output,
                "perturbed_outputs": ex.perturbed_outputs,
                "is_perturbed": bool(ex.metadata.get("is_perturbed", False)),
                "metadata": ex.metadata,
                **{k: v for k, v in {
                    "consistency": ex.consistency,
                    "accuracy": ex.accuracy,
                }.items() if v is not None},
            }
            for ex in examples
        ],
        "result_trajectory": "trajectory.json",
    }

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    trajectory_payload = {
        "created_at": timestamp.isoformat(),
        "status": "in_progress",
        "trajectory_count": len(trajectories),
        "entries": [asdict(entry) for entry in trajectories],
    }
    trajectory_path = run_dir / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(trajectory_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# Evaluation loop
def evaluate(
    items: List[BenchmarkItem],
    agent_fn: Callable[[str, BaseEnvironment], AgentRunResult],
    benchmark: BaseBenchmark,
    config: EvalConfig,
    perturb_fns: List[Tuple[Callable[[str], str], Dict[str, Any]]],
    run_dir: Optional[Path] = None,
    timestamp: Optional[datetime] = None,
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
                    metadata={**item.metadata, **dict(base_result.metadata)},
                )
            )
        else:
            base_output = str(base_result)
        perts = generate_perturbations(
            item.prompt,
            perturb_instances,
            n=config.n_perturbations,
            seed=config.seed,
        )
        perturbed_outputs: List[Dict[str, Any]] = []
        for p_type, p_text, p_inst in perts:
            if getattr(p_inst, 'modifies_code', False):
                base_commit = getattr(item, 'base_commit', None)
                if hasattr(p_inst, '_apply_to_env'):
                    p_inst._apply_to_env(item.env, base_commit=base_commit)

            pert_result = agent_fn(p_text, item.env)
            if isinstance(pert_result, AgentRunResult):
                out = pert_result.output
                pert_metadata = dict(pert_result.metadata)
                pert_metadata.setdefault("perturbation", p_type)
                pert_metadata = {**item.metadata, **pert_metadata}
                if getattr(p_inst, 'modifies_code', False):
                    pert_metadata["code_modification"] = getattr(p_inst, 'last_result', {})
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
                metadata=dict(item.metadata),
                **item_score,
            )
        )

        # Save results incrementally after each example
        if run_dir is not None:
            _save_incremental_results(
                run_dir=run_dir,
                config=asdict(config),
                examples=examples,
                trajectories=trajectories,
                benchmark=benchmark,
                timestamp=timestamp or datetime.now(),
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
    perturb_instances  = _instantiate_perturbations(perturb_cfgs)

    # Setup output directory early for incremental saves
    out_path_str = raw_cfg.get("output", {}).get("path") or raw_cfg.get("out")
    run_dir = None
    timestamp = datetime.now()

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

        run_dir_name = f"{base_name}-{timestamp.strftime('%Y-%m-%d::%H-%M-%S')}"
        run_dir = base_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results incrementally to: {run_dir}")

    # Evaluate with incremental saving
    result = evaluate(
        items, agent_fn, benchmark, eval_cfg, perturb_fns,
        run_dir=run_dir,
        timestamp=timestamp,
    )

    # Final save with completed status
    payload = {
        "config": result.config,
        "status": "completed",
        **{k: v for k, v in {
            "total": result.total,
            "consistency": result.consistency,
            "accuracy": result.accuracy,
        }.items() if v is not None},
        "examples": [
            {
                "id": ex.id,
                "base_prompt": ex.base_prompt,
                "base_output": ex.base_output,
                "perturbed_outputs": ex.perturbed_outputs,
                "is_perturbed": bool(ex.metadata.get("is_perturbed", False)),
                "metadata": ex.metadata,
                **{k: v for k, v in {
                    "consistency": ex.consistency,
                    "accuracy": ex.accuracy,
                }.items() if v is not None},
            }
            for ex in result.examples
        ],
    }

    if run_dir:
        trajectory_filename = "trajectory.json"
        trajectory_path = run_dir / trajectory_filename
        result_path = run_dir / "result.json"

        payload["result_trajectory"] = trajectory_filename

        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        trajectory_payload = {
            "created_at": timestamp.isoformat(),
            "status": "completed",
            "results_dir": run_dir.name,
            "trajectory_file": trajectory_filename,
            "trajectory_count": len(result.trajectories),
            "entries": [asdict(entry) for entry in result.trajectories],
        }
        trajectory_path.write_text(
            json.dumps(trajectory_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print(f"Wrote final results to {result_path} and trajectory log to {trajectory_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
