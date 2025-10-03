from __future__ import annotations

import yaml
import importlib
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Data models for results
# -----------------------------


@dataclass
class BenchmarkItem:
    id: str
    prompt: str
    label: Optional[str] = None


@dataclass
class EvalConfig:
    n_perturbations: int = 5
    seed: int = 42


@dataclass
class ExampleResult:
    id: str
    base_output: str
    perturbed_outputs: List[Dict[str, Any]]
    consistent: bool


@dataclass
class EvalResult:
    config: Dict[str, Any]
    total: int
    consistent: int
    consistency_rate: float
    examples: List[ExampleResult]


# -----------------------------
# Helpers
# -----------------------------


def _resolve_object(dotted: str) -> Any:
    mod_path, _, attr = dotted.partition(":")
    if not attr:
        raise ValueError("Expected dotted path in format 'module.sub:attr'")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _maybe_instantiate(obj: Any, params: Optional[Dict[str, Any]] = None) -> Any:
    if isinstance(obj, type):
        return obj(**(params or {}))
    return obj


def _normalize_text(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def compute_consistency(base: str, perturbed: List[str]) -> Tuple[bool, List[bool]]:
    base_n = _normalize_text(base)
    flags = [(_normalize_text(p) == base_n) for p in perturbed]
    return all(flags), flags


# -----------------------------
# Benchmark loading
# -----------------------------


def _iter_benchmark_examples(benchmark_obj: Any) -> Iterable[Dict[str, Any]]:
    # Support either .iter() or __iter__ on benchmark
    if hasattr(benchmark_obj, "iter") and callable(getattr(benchmark_obj, "iter")):
        return benchmark_obj.iter()
    if hasattr(benchmark_obj, "__iter__"):
        return iter(benchmark_obj)
    raise TypeError("Benchmark object must define an `iter()` or `__iter__` method")


def load_benchmark_from_config(bm_cfg: Dict[str, Any]) -> List[BenchmarkItem]:
    """Create a benchmark instance from config and return items."""
    path = bm_cfg.get("path")
    params = bm_cfg.get("params", {})
    if not path:
        raise ValueError("benchmark.path must be provided in config.yaml")

    obj = _resolve_object(path)
    benchmark = _maybe_instantiate(obj, params)
    if hasattr(benchmark, "load") and callable(getattr(benchmark, "load")):
        benchmark.load()

    items: List[BenchmarkItem] = []
    for i, ex in enumerate(_iter_benchmark_examples(benchmark)):
        # Expect examples to include a prompt-like field
        prompt = (
            ex.get("prompt")
            or ex.get("input")
            or ex.get("text")
            or ex.get("question")
        )
        if prompt is None:
            continue
        ex_id = str(ex.get("id") or i)
        label = ex.get("label") if isinstance(ex.get("label"), (str, int)) else None
        items.append(BenchmarkItem(id=ex_id, prompt=str(prompt), label=str(label) if label is not None else None))
    return items


# -----------------------------
# Perturbations
# -----------------------------


def _instantiate_perturbations(cfg_list: List[Dict[str, Any]]) -> List[Callable[[str], str]]:
    perts: List[Callable[[str], str]] = []
    for pcfg in cfg_list:
        path = pcfg.get("path")
        params = pcfg.get("params", {})
        if not path:
            continue
        obj = _resolve_object(path)
        inst = _maybe_instantiate(obj, params)
        if hasattr(inst, "apply") and callable(getattr(inst, "apply")):
            perts.append(lambda text, inst=inst: inst.apply(text))
        elif callable(inst):
            perts.append(inst)
        else:
            raise TypeError(f"Perturbation at {path} must be callable or implement .apply(text)")
    return perts


def generate_perturbations(
    text: str,
    perturb_fns: List[Callable[[str], str]],
    n: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """Return list of (name, perturbed_text) for the given input text."""
    rng = random.Random(seed)
    if not perturb_fns:
        return []
    results: List[Tuple[str, str]] = []
    for i in range(n):
        fn = perturb_fns[i % len(perturb_fns)]
        name = getattr(fn, "__name__", getattr(getattr(fn, "__self__", object()), "__class__", type("_", (), {})).__name__)
        perturbed = fn(text)
        if rng.random() < -1.0:  # reserved hook for future randomization
            pass
        results.append((name, perturbed))
    return results


# -----------------------------
# Agent resolution
# -----------------------------


def resolve_agent_callable(cfg: Dict[str, Any]) -> Callable[[str], str]:
    """Resolve agent from config into a callable(prompt) -> str.

    Supports either a function or a class with a `run(task: str) -> Any` method.
    If model/environment are configured, they will be instantiated and passed
    to the agent class constructor.
    """
    path = cfg.get("path")
    params = cfg.get("params", {})
    if not path:
        raise ValueError("agent.path must be provided in config.yaml")

    obj = _resolve_object(path)

    # Direct callable (function) or simple builtin types like `str`
    simple_builtin_types = (str, int, float, bool)
    if callable(obj) and (not isinstance(obj, type) or obj in simple_builtin_types):
        return lambda prompt: str(obj(prompt, **params))  # type: ignore[misc]

    # Class with run(str)->Any
    if isinstance(obj, type):
        # Optional model/environment
        model = None
        env = None
        if cfg.get("model"):
            mpath = cfg["model"].get("path")
            mparams = cfg["model"].get("params", {})
            if mpath:
                model = _maybe_instantiate(_resolve_object(mpath), mparams)
        if cfg.get("environment"):
            epath = cfg["environment"].get("path")
            eparams = cfg["environment"].get("params", {})
            if epath:
                env = _maybe_instantiate(_resolve_object(epath), eparams)

        try:
            if model is not None or env is not None:
                instance = obj(model=model, env=env, **params)
            else:
                instance = obj(**params)
        except TypeError:
            # Fallback to zero-arg instantiation
            instance = obj()

        if hasattr(instance, "run") and callable(getattr(instance, "run")):
            def _runner(prompt: str) -> str:
                res = instance.run(prompt)
                if isinstance(res, tuple) and res:
                    return str(res[-1])
                return str(res)

            return _runner

    raise TypeError(
        "Agent must refer to a function(prompt)->str or a class with run(str)->Any"
    )


# -----------------------------
# Evaluation loop
# -----------------------------


def evaluate(
    items: List[BenchmarkItem],
    agent_fn: Callable[[str], str],
    config: EvalConfig,
    perturb_fns: List[Callable[[str], str]],
) -> EvalResult:
    examples: List[ExampleResult] = []
    consistent_count = 0

    for item in items:
        base_output = agent_fn(item.prompt)
        perts = generate_perturbations(
            item.prompt,
            perturb_fns,
            n=config.n_perturbations,
            seed=config.seed,
        )
        perturbed_outputs: List[Dict[str, Any]] = []
        for p_type, p_text in perts:
            out = agent_fn(p_text)
            perturbed_outputs.append({
                "type": p_type,
                "text": p_text,
                "output": out,
            })

        ok, flags = compute_consistency(
            base_output, [po["output"] for po in perturbed_outputs]
        )
        for f, po in zip(flags, perturbed_outputs):
            po["equal_to_base"] = f

        if ok:
            consistent_count += 1

        examples.append(
            ExampleResult(
                id=item.id,
                base_output=base_output,
                perturbed_outputs=perturbed_outputs,
                consistent=ok,
            )
        )

    total = len(items)
    rate = (consistent_count / total) if total else 0.0
    return EvalResult(
        config=asdict(config),
        total=total,
        consistent=consistent_count,
        consistency_rate=rate,
        examples=examples,
    )


def _load_config(config_path: str | Path) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: Optional[List[str]] = None) -> int:
    # Expect config path as first arg or default to ./config.yaml
    cfg_path = Path(argv[0]) if argv else Path("config.yaml")
    raw_cfg = _load_config(cfg_path)
    eval_cfg = EvalConfig(
        n_perturbations=int(raw_cfg.get("eval", {}).get("n_perturbations", 5)),
        seed=int(raw_cfg.get("eval", {}).get("seed", 42)),
    )

    # Benchmark
    items = load_benchmark_from_config(raw_cfg.get("benchmark", {}))

    # Agent
    agent_fn = resolve_agent_callable(raw_cfg.get("agent", {}))

    # Perturbations
    perturb_cfgs = raw_cfg.get("perturbations", [])
    if isinstance(perturb_cfgs, dict):
        perturb_cfgs = [perturb_cfgs]
    perturb_fns = _instantiate_perturbations(perturb_cfgs)

    # Evaluate
    result = evaluate(items, agent_fn, eval_cfg, perturb_fns)

    payload = {
        "config": result.config,
        "total": result.total,
        "consistent": result.consistent,
        "consistency_rate": result.consistency_rate,
        "examples": [
            {
                "id": ex.id,
                "base_output": ex.base_output,
                "perturbed_outputs": ex.perturbed_outputs,
                "consistent": ex.consistent,
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
