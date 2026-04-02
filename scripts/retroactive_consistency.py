#!/usr/bin/env python3
"""
Retroactively compute per-example binary consistency from existing result.json files.

Consistency = 1.0 if base and all perturbed runs agree (all pass or all fail), else 0.0.
For Harbor benchmarks, success is determined by verifier reward > 0.

Handles two result formats:
  1. Paired: each example has base_output + perturbed_outputs list (SWE-bench, BFCL)
  2. Separate rows: base and perturbed are separate examples with is_perturbed flag,
     matched by ID (Spider2-DBT: "taskid" vs "taskid_perturbed")

Usage:
    python scripts/retroactive_consistency.py outputs/harbor-spider2-dbt-oracle-header-shuffle
    python scripts/retroactive_consistency.py   # all output dirs
"""

import json
import glob
import re
import sys
from pathlib import Path


def is_success(text: str) -> bool:
    try:
        return float(text) > 0.0
    except (TypeError, ValueError):
        return False


def compute_consistency_paired(examples):
    """Compute consistency from paired format (base_output + perturbed_outputs)."""
    consistency_scores = []
    base_correct = 0
    pert_correct = 0
    base_total = 0
    pert_total = 0

    for ex in examples:
        base_output = ex.get("base_output", "")
        perturbed_outputs = ex.get("perturbed_outputs", [])

        if not perturbed_outputs:
            continue

        base_pass = is_success(base_output)
        pert_passes = []
        for p in perturbed_outputs:
            po = p.get("output", "") if isinstance(p, dict) else str(p)
            pert_passes.append(is_success(po))

        all_results = [base_pass] + pert_passes
        consistent = 1.0 if len(set(all_results)) == 1 else 0.0
        consistency_scores.append(consistent)

        base_total += 1
        base_correct += int(base_pass)
        for pp in pert_passes:
            pert_total += 1
            pert_correct += int(pp)

        ex["consistency"] = consistent

    return consistency_scores, base_correct, base_total, pert_correct, pert_total


def compute_consistency_separate(examples):
    """Compute consistency from separate rows (is_perturbed flag, matched by ID)."""
    # Separate base and perturbed examples
    base_map = {}  # base_id -> example
    pert_map = {}  # base_id -> [perturbed examples]

    for ex in examples:
        eid = ex.get("id", "")
        is_pert = ex.get("is_perturbed", False)

        if is_pert:
            # Strip _perturbed suffix to get base ID
            base_id = re.sub(r"_perturbed$", "", eid)
            pert_map.setdefault(base_id, []).append(ex)
        else:
            base_map[eid] = ex

    consistency_scores = []
    base_correct = 0
    pert_correct = 0
    base_total = 0
    pert_total = 0

    for base_id, base_ex in base_map.items():
        perts = pert_map.get(base_id, [])
        if not perts:
            continue

        base_pass = is_success(base_ex.get("base_output", ""))
        pert_passes = [is_success(p.get("base_output", "")) for p in perts]

        all_results = [base_pass] + pert_passes
        consistent = 1.0 if len(set(all_results)) == 1 else 0.0
        consistency_scores.append(consistent)

        base_total += 1
        base_correct += int(base_pass)
        for pp in pert_passes:
            pert_total += 1
            pert_correct += int(pp)

        # Tag examples with consistency
        base_ex["consistency"] = consistent
        for p in perts:
            p["consistency"] = consistent

    return consistency_scores, base_correct, base_total, pert_correct, pert_total


def compute_consistency(result_path: Path) -> dict:
    with open(result_path) as f:
        data = json.load(f)

    examples = data.get("examples", [])
    if not examples:
        return {"consistency": None, "n": 0}

    # Detect format: check if any example has non-empty perturbed_outputs
    has_paired = any(ex.get("perturbed_outputs") for ex in examples)
    has_separate = any(ex.get("is_perturbed") is not None for ex in examples)

    if has_paired:
        scores, bc, bt, pc, pt = compute_consistency_paired(examples)
    elif has_separate:
        scores, bc, bt, pc, pt = compute_consistency_separate(examples)
    else:
        return {"consistency": None, "n": 0}

    if not scores:
        return {"consistency": None, "n": 0}

    overall = sum(scores) / len(scores)
    data["consistency"] = overall

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return {
        "consistency": overall,
        "n": len(scores),
        "base_acc": bc / bt if bt else None,
        "pert_acc": pc / pt if pt else None,
    }


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob("outputs/harbor-*"))

    for path in paths:
        p = Path(path)
        candidates = sorted(p.glob("**/result.json"), reverse=True)
        if not candidates:
            continue

        result_file = candidates[0]
        result = compute_consistency(result_file)

        if result["consistency"] is not None:
            print(
                f"{p.name:55s} | consistency={result['consistency']:.3f} "
                f"| base_acc={result['base_acc']:.3f} "
                f"| pert_acc={result['pert_acc']:.3f} "
                f"| n={result['n']}"
            )
        else:
            print(f"{p.name:55s} | no perturbations / baseline only")


if __name__ == "__main__":
    main()
