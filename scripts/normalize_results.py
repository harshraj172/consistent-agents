"""
Normalize all results to 1 perturbation per example. Use a single canonical
baseline per group and recompute consistency of each perturbation run against
that fixed baseline.

Canonical baseline selection:
  - For groups with a dedicated baseline run, use that.
  - For groups where every run re-evaluated the base independently, pick the
    first listed run's base results as canonical.
  - For Spider2-DBT, the baseline run (if present) is used; otherwise the
    first perturbation run's base results.

Consistency = 1.0 if canonical_base and pert agree (both pass or both fail).

Note: linear_mcp for codex-gpt5mini has a fundamentally different base setup
(MCP tools in instruction), so it gets its own baseline — not shared with
injection_swe.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def is_success(output: str) -> bool:
    """Check if a Harbor output indicates success (reward > 0)."""
    try:
        return float(output) > 0.0
    except (TypeError, ValueError):
        return False


def load_result(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _prompt_key(prompt: str) -> str:
    """Stable key from prompt (first 200 chars, stripped)."""
    return prompt.strip()[:200]


def extract_base_map(
    data: Dict[str, Any],
    is_spider2: bool = False,
    key_by: str = "id",
) -> Dict[str, str]:
    """Extract {key: base_output} from a result.json.

    key_by: "id" uses example ID, "prompt" uses base_prompt (for cross-run matching).
    """
    base_map = {}
    for ex in data.get("examples", []):
        if is_spider2 and str(ex["id"]).endswith("_perturbed"):
            continue
        if key_by == "prompt":
            key = _prompt_key(ex.get("base_prompt", ""))
        else:
            key = str(ex["id"])
        base_map[key] = str(ex.get("base_output", ""))
    return base_map


def extract_pert_map(
    data: Dict[str, Any],
    is_spider2: bool = False,
    key_by: str = "id",
) -> Dict[str, str]:
    """Extract {key: first_pert_output} from a result.json.

    key_by: "id" uses example ID, "prompt" uses base_prompt (for cross-run matching).
    """
    pert_map = {}
    if is_spider2:
        for ex in data.get("examples", []):
            eid = str(ex["id"])
            if eid.endswith("_perturbed"):
                base_id = eid.replace("_perturbed", "")
                if key_by == "prompt":
                    # For spider2, perturbed examples don't have a base_prompt to match on
                    # Use the base_id as key
                    pert_map[base_id] = str(ex.get("base_output", ""))
                else:
                    pert_map[base_id] = str(ex.get("base_output", ""))
    else:
        for ex in data.get("examples", []):
            if key_by == "prompt":
                key = _prompt_key(ex.get("base_prompt", ""))
            else:
                key = str(ex["id"])
            perts = ex.get("perturbed_outputs", [])
            if perts:
                pert_map[key] = str(perts[0].get("output", ""))
    return pert_map


def compute_consistency(
    canonical_base: Dict[str, str],
    pert_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Compute per-example consistency between canonical base and perturbation outputs.
    Returns list of {id, base_output, pert_output, base_pass, pert_pass, consistency}.
    """
    results = []
    for eid, base_output in canonical_base.items():
        pert_output = pert_map.get(eid)
        if pert_output is None:
            continue
        base_pass = is_success(base_output)
        pert_pass = is_success(pert_output)
        results.append({
            "id": eid,
            "base_output": base_output,
            "pert_output": pert_output,
            "base_pass": base_pass,
            "pert_pass": pert_pass,
            "consistency": 1.0 if base_pass == pert_pass else 0.0,
        })
    return results


def aggregate_stats(per_example: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate stats from per-example consistency results."""
    if not per_example:
        return {"base_accuracy": 0.0, "pert_accuracy": 0.0, "consistency": None, "total": 0}

    n = len(per_example)
    base_passes = sum(1 for e in per_example if e["base_pass"])
    pert_passes = sum(1 for e in per_example if e["pert_pass"])
    consistencies = [e["consistency"] for e in per_example]

    return {
        "base_accuracy": base_passes / n,
        "pert_accuracy": pert_passes / n,
        "consistency": sum(consistencies) / len(consistencies),
        "total": n,
    }


def normalize_all(
    src_dir: str = "results_to_upload",
    dst_dir: str = "results_normalized",
) -> Dict[str, Any]:
    src = Path(src_dir)
    dst = Path(dst_dir)

    # Define groups.
    # Each group has: canonical_baseline_run, and list of perturbation runs.
    # canonical_baseline_run: which run's base results to use as the fixed baseline.
    #   - None means use the first run's base (for groups without a dedicated baseline).
    groups = {
        "swebench/codex-gpt5mini": {
            "baseline_from": "swebench/codex-gpt5mini/injection_swe",
            "id_mapping": "prompt",  # match by prompt since IDs differ across runs
            "runs": [
                ("injection_swe", "swebench/codex-gpt5mini/injection_swe"),
                ("linear_mcp", "swebench/codex-gpt5mini/linear_mcp"),
            ],
            "is_spider2": False,
        },
        "swebench/openhands-kimik2": {
            "baseline_from": "swebench/openhands-kimik2/noise",
            "runs": [
                ("noise", "swebench/openhands-kimik2/noise"),
                ("paraphrase", "swebench/openhands-kimik2/paraphrase"),
                ("translation", "swebench/openhands-kimik2/translation"),
                ("injection_swe", "swebench/openhands-kimik2/injection_swe"),
                ("linear_mcp", "swebench/openhands-kimik2/linear_mcp"),
            ],
            "is_spider2": False,
        },
        "spider2-dbt/codex-gpt5mini": {
            "baseline_from": "spider2-dbt/codex-gpt5mini/baseline",
            "runs": [
                ("timestamp", "spider2-dbt/codex-gpt5mini/timestamp"),
                ("header_shuffle", "spider2-dbt/codex-gpt5mini/header_shuffle"),
                ("header_translate", "spider2-dbt/codex-gpt5mini/header_translate"),
            ],
            "is_spider2": True,
        },
        "spider2-dbt/openhands-kimik2": {
            "baseline_from": "spider2-dbt/openhands-kimik2/header_translate",
            "runs": [
                ("header_translate", "spider2-dbt/openhands-kimik2/header_translate"),
            ],
            "is_spider2": True,
        },
    }

    summary = {}

    for group_key, group_cfg in groups.items():
        is_spider2 = group_cfg["is_spider2"]
        baseline_rel = group_cfg["baseline_from"]

        # Load canonical baseline
        baseline_path = src / baseline_rel / "result.json"
        if not baseline_path.exists():
            print(f"SKIP {group_key}: baseline not found at {baseline_path}")
            continue

        key_by = group_cfg.get("id_mapping", "id")
        baseline_data = load_result(str(baseline_path))
        canonical_base = extract_base_map(baseline_data, is_spider2=is_spider2, key_by=key_by)

        base_passes = sum(1 for v in canonical_base.values() if is_success(v))
        base_acc = base_passes / len(canonical_base) if canonical_base else 0.0
        print(f"{group_key}: canonical baseline from {baseline_rel} = {base_passes}/{len(canonical_base)} = {base_acc:.3f} (key_by={key_by})")

        group_summary = {
            "canonical_baseline": baseline_rel,
            "base_accuracy": round(base_acc, 4),
            "runs": {},
        }

        for pert_name, rel_path in group_cfg["runs"]:
            result_path = src / rel_path / "result.json"
            if not result_path.exists():
                print(f"  SKIP {pert_name}: not found")
                continue

            data = load_result(str(result_path))
            pert_map = extract_pert_map(data, is_spider2=is_spider2, key_by=key_by)

            # Compute consistency against canonical baseline
            per_example = compute_consistency(canonical_base, pert_map)
            stats = aggregate_stats(per_example)

            # Write output
            out_dir = dst / group_key / pert_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_data = {
                "group": group_key,
                "perturbation": pert_name,
                "canonical_baseline": baseline_rel,
                "base_accuracy": round(base_acc, 4),
                "pert_accuracy": round(stats["pert_accuracy"], 4),
                "consistency": round(stats["consistency"], 4) if stats["consistency"] is not None else None,
                "total": stats["total"],
                "examples": per_example,
            }

            with open(out_dir / "result.json", "w") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2, default=str)

            group_summary["runs"][pert_name] = {
                "pert_accuracy": round(stats["pert_accuracy"], 4),
                "consistency": round(stats["consistency"], 4) if stats["consistency"] is not None else None,
                "total": stats["total"],
            }

            print(f"  {pert_name}: pert_acc={stats['pert_accuracy']:.3f}, consistency={stats['consistency']:.3f}, n={stats['total']}")

        summary[group_key] = group_summary

    return summary


if __name__ == "__main__":
    summary = normalize_all()
    print("\n" + json.dumps(summary, indent=2))
