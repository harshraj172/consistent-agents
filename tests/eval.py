from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from consistent_agents.data_models import EvalConfig
from consistent_agents.eval import (
    _instantiate_perturbations,
    _load_config,
    generate_perturbations,
    load_benchmark_from_config,
    resolve_agent_callable,
)


def _prepare_perturbations(raw_cfg: Dict[str, Any]) -> List[Tuple]:
    perturb_cfgs = raw_cfg.get("perturbations", [])
    if isinstance(perturb_cfgs, dict):
        perturb_cfgs = [perturb_cfgs]
    perturb_fns = _instantiate_perturbations(perturb_cfgs)
    return [(fn, cfg) for fn, cfg in zip(perturb_fns, perturb_cfgs)]


def _build_eval_config(raw_cfg: Dict[str, Any]) -> EvalConfig:
    eval_cfg = raw_cfg.get("eval", {})
    return EvalConfig(
        n_perturbations=int(eval_cfg.get("n_perturbations", 5)),
        seed=int(eval_cfg.get("seed", 42)),
    )


def _run_single_turn_eval(cfg_path: Path) -> Dict[str, Any]:
    raw_cfg = _load_config(cfg_path)
    eval_cfg = _build_eval_config(raw_cfg)
    benchmark, items = load_benchmark_from_config(raw_cfg.get("benchmark", {}))
    agent_fn = resolve_agent_callable(raw_cfg.get("agent", {}))
    perturbations = _prepare_perturbations(raw_cfg)

    consistency_total = 0
    accuracy_total = 0
    total = 0
    example_payloads: List[Dict[str, Any]] = []

    for item in items:
        base_output = agent_fn(item.prompt, item.env)
        generated_perts = generate_perturbations(
            item.prompt,
            perturbations,
            n=eval_cfg.n_perturbations,
            seed=eval_cfg.seed,
        )
        perturbed_outputs: List[Dict[str, Any]] = []
        for pert_type, pert_prompt in generated_perts:
            pert_output = agent_fn(pert_prompt, item.env)
            perturbed_outputs.append(
                {
                    "type": pert_type,
                    "text": pert_prompt,
                    "output": pert_output,
                }
            )
        scores = benchmark.score(
            item.id,
            base_output,
            [entry["output"] for entry in perturbed_outputs],
        )
        row_total = scores["total"] or 1
        consistency_total += scores["consistent_count"]
        accuracy_total += scores["correct_count"]
        total += scores["total"]

        example_payloads.append(
            {
                "id": item.id,
                "base_output": base_output,
                "perturbed_outputs": perturbed_outputs,
                "consistency": scores["consistent_count"] / row_total,
                "accuracy": scores["correct_count"] / row_total,
            }
        )

    return {
        "config": {
            "n_perturbations": eval_cfg.n_perturbations,
            "seed": eval_cfg.seed,
        },
        "total": total,
        "consistency": consistency_total / total if total else 0.0,
        "accuracy": accuracy_total / total if total else 0.0,
        "examples": example_payloads,
    }


def _load_baseline(raw_cfg: Dict[str, Any], baseline_override: str | None) -> Dict[str, Any]:
    baseline_path = baseline_override or raw_cfg.get("output", {}).get("path") or raw_cfg.get("out")
    if not baseline_path:
        raise ValueError("No baseline output path found in config or CLI arguments.")
    path = Path(baseline_path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_results(
    manual: Dict[str, Any], baseline: Dict[str, Any], tolerance: float
) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    ok = True

    for key in ("total", "consistency", "accuracy"):
        manual_value = manual[key]
        baseline_value = baseline[key]
        if key == "total":
            match = manual_value == baseline_value
        else:
            match = math.isclose(float(manual_value), float(baseline_value), rel_tol=tolerance, abs_tol=tolerance)
        if not match:
            ok = False
            diff = float(manual_value) - float(baseline_value)
            messages.append(f"{key} mismatch: manual={manual_value} baseline={baseline_value} diff={diff}")

    baseline_examples = {ex["id"]: ex for ex in baseline.get("examples", [])}
    for example in manual.get("examples", []):
        base = baseline_examples.get(example["id"])
        if not base:
            ok = False
            messages.append(f"Example {example['id']} missing in baseline")
            continue
        for field in ("consistency", "accuracy"):
            manual_val = float(example[field])
            base_val = float(base[field])
            if not math.isclose(manual_val, base_val, rel_tol=tolerance, abs_tol=tolerance):
                ok = False
                diff = manual_val - base_val
                messages.append(
                    f"Example {example['id']} {field} mismatch: manual={manual_val} baseline={base_val} diff={diff}"
                )

    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Parity check for TruthfulQA evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tests/config.yaml"),
        help="Path to evaluation config.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Override path to baseline JSON output for comparison.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for floating point comparison.",
    )
    parser.add_argument(
        "--dump-manual",
        type=Path,
        default=None,
        help="Optional path to write the manual evaluation payload.",
    )
    args = parser.parse_args()

    raw_cfg = _load_config(args.config)
    manual_result = _run_single_turn_eval(args.config)
    baseline_result = _load_baseline(raw_cfg, args.baseline)

    ok, messages = _compare_results(manual_result, baseline_result, args.tolerance)

    print("Manual metrics:")
    print(json.dumps({k: manual_result[k] for k in ("total", "consistency", "accuracy")}, indent=2))
    print("Baseline metrics:")
    print(json.dumps({k: baseline_result[k] for k in ("total", "consistency", "accuracy")}, indent=2))

    if args.dump_manual:
        args.dump_manual.parent.mkdir(parents=True, exist_ok=True)
        args.dump_manual.write_text(json.dumps(manual_result, indent=2), encoding="utf-8")
        print(f"Wrote manual evaluation payload to {args.dump_manual}")

    if ok:
        print("Parity check passed.")
        return 0

    print("Parity check failed:")
    for line in messages:
        print(f"- {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
