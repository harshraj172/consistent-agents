from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from datasets import concatenate_datasets, Dataset


@dataclass
class BenchmarkItem:
    id: str
    input: str
    label: Optional[str] = None


@dataclass
class EvalConfig:
    n_perturbations: int = 5
    perturbation_types: Tuple[str, ...] = (
        "whitespace",
        "punctuation",
        "case",
        "synonym",
        "stopword_shuffle",
    )
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


def load_benchmark(path: str | Path) -> List[BenchmarkItem]:    



def generate_perturbations(
    dataset: Dataset,
    types: Iterable[str],
    batch_size: int = 64,
    n: int = 1,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Return list of (type, perturbed_text) for the given input text."""
    types = list(types)
    num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)
    perturbed_dataset = None
    for i in range(n):
        t = types[i % len(types)]
        fn = PERTURB_FN.get(t)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))
            ds = dataset.select(range(start_idx, end_idx))
            perturbed_ds = fn(ds, random.Random(seed))
            if perturbed_dataset is None:
                perturbed_dataset = perturbed_ds
            else:
                perturbed_dataset = concatenate_datasets([perturbed_dataset, perturbed_ds])
    return perturbed_dataset


def _resolve_object(dotted: str) -> Any:
    mod_path, _, attr = dotted.partition(":")
    if not attr:
        raise ValueError("Expected dotted path in format 'module.sub:attr'")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def resolve_agent_callable(agent_path: str) -> Callable[[str], str]:
    """Resolve a user-provided path into a callable(prompt) -> str.

    Accepts either a function or a class with a zero-arg constructor
    exposing a `run(task: str) -> str` method.
    """
    obj = _resolve_object(agent_path)

    # Direct function
    if callable(obj) and not isinstance(obj, type):
        return obj  # type: ignore[return-value]

    # Class with run(str)->str
    if isinstance(obj, type):
        try:
            instance = obj()  # zero-arg constructor for stub
        except Exception as e:  # noqa: BLE001 - bubble up with context
            raise TypeError(
                f"Failed to instantiate agent class {obj.__name__} with zero-arg constructor: {e}"
            ) from e

        if hasattr(instance, "run") and callable(getattr(instance, "run")):
            def _runner(prompt: str) -> str:
                return str(instance.run(prompt))

            return _runner

    raise TypeError(
        "Agent path must refer to a function(prompt)->str or a class with run(str)->str"
    )


# -----------------------------
# Normalization and metrics
# -----------------------------


def normalize_output(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_consistency(base: str, others: Iterable[str]) -> Tuple[bool, List[bool]]:
    base_n = normalize_output(base)
    flags: List[bool] = []
    for o in others:
        flags.append(normalize_output(o) == base_n)
    return (all(flags) if flags else True, flags)


# -----------------------------
# Evaluation loop
# -----------------------------


def evaluate(
    items: List[BenchmarkItem],
    agent_fn: Callable[[str], str],
    config: EvalConfig,
) -> EvalResult:
    examples: List[ExampleResult] = []
    consistent_count = 0

    for item in items:
        base_output = agent_fn(item.input)
        perts = generate_perturbations(
            item.input,
            n=config.n_perturbations,
            types=config.perturbation_types,
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


# -----------------------------
# CLI
# -----------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate agent consistency under perturbations")
    p.add_argument("benchmark", type=str, help="Path to JSON/JSONL file or directory")
    p.add_argument("agent", type=str, help="Dotted path to agent function or class, e.g. pkg.mod:run")
    p.add_argument("--n-perturbations", type=int, default=5)
    p.add_argument(
        "--types",
        type=str,
        default="whitespace,punctuation,case,synonym,stopword_shuffle",
        help="Comma-separated perturbation types",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    items = load_benchmark(args.benchmark)
    agent_fn = resolve_agent_callable(args.agent)
    cfg = EvalConfig(
        n_perturbations=args.n_perturbations,
        perturbation_types=tuple(t.strip() for t in args.types.split(",") if t.strip()),
        seed=args.seed,
    )
    result = evaluate(items, agent_fn, cfg)

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

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote results to {out_path}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
