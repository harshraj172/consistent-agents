#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import math
from scipy.spatial.distance import jensenshannon

CONSTANT_MULTIPLIER = 3.0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _extract_actions_from_atif_steps(
    steps: Sequence[Dict[str, Any]],
) -> List[str]:
    """
    Action ontology used here:
      - each tool invocation becomes one action token: tool_calls[*].function_name

    We intentionally exclude "system" and "user" sources (prompt/context).
    """
    actions: List[str] = []
    for step in steps:
        if step.get("source") != "agent":
            continue

        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        for call in tool_calls:
            if not isinstance(call, dict):
                continue

            fn = call.get("function_name")
            if not isinstance(fn, str) or not fn:
                continue

            actions.append(fn)
    return actions


def _action_sequence(entry: Dict[str, Any]) -> List[str]:
    """
    Extract actions ONLY from this run's trajectory.json (entries[].messages).
    """
    msgs = entry.get("messages")
    return _extract_actions_from_atif_steps(msgs) if isinstance(msgs, list) else []


def _seqs_to_prob_matrix(seqs: Sequence[Sequence[str]]) -> np.ndarray:
    vocab: Dict[str, int] = {}
    for s in seqs:
        for tok in s:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    if not vocab:
        return np.zeros((len(seqs), 0), dtype=np.float64)

    mat = np.zeros((len(seqs), len(vocab)), dtype=np.float64)
    for i, s in enumerate(seqs):
        for tok in s:
            mat[i, vocab[tok]] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    return mat


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    sp = float(p.sum())
    sq = float(q.sum())

    if sp == 0.0 and sq == 0.0:
        return 0.0
    if sp == 0.0 or sq == 0.0:
        return 1.0

    d = float(jensenshannon(p, q, base=2.0))
    return d * d


def _levenshtein_distance(
    a: Sequence[str],
    b: Sequence[str],
    *,
    mode: str = "unweighted",
) -> float:
    if mode != "unweighted":
        return _weighted_levenshtein(a, b, mode=mode)

    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))

    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [float(i)]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1.0
            dele = prev[j] + 1.0
            sub = prev[j - 1] + (0.0 if ca == cb else 1.0)
            curr.append(min(ins, dele, sub))
        prev = curr

    return float(prev[-1])


def _weighted_levenshtein(
    s1: Sequence[str], s2: Sequence[str], mode: str = "exponential"
) -> float:
    m, n = len(s1), len(s2)

    if m == 0:
        return float(n)
    if n == 0:
        return float(m)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    def weight(i, j):
        # normalized positions
        wi = i / m
        wj = j / n
        base = (wi + wj) / 2

        if mode == "linear":
            return base
        elif mode == "constant":
            return CONSTANT_MULTIPLIER * base
        elif mode == "exponential":
            return math.exp(base)  # can replace with exp(c * base)
        else:
            raise ValueError("Invalid mode")

    # init
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + weight(i, 0)

    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + weight(0, j)

    # DP
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            w = weight(i, j)

            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = w

            dp[i][j] = min(
                dp[i - 1][j] + w,  # deletion
                dp[i][j - 1] + w,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def _max_weight_for_mode(mode: str) -> float:
    if mode == "unweighted":
        return 1.0

    weighted_mode = mode.removeprefix("weighted_")

    if weighted_mode == "linear":
        return 1.0

    if weighted_mode == "constant":
        return CONSTANT_MULTIPLIER

    if weighted_mode == "exponential":
        return math.exp(1.0)

    raise ValueError(f"Invalid mode: {mode}")


def _normalization_denominator(
    a: Sequence[str],
    b: Sequence[str],
    *,
    mode: str,
) -> float:
    return float(max(len(a), len(b))) * _max_weight_for_mode(mode)


"""
def weighted_levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize matrix
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    # Weight function: Higher i/j = higher cost
    # We use (i/m) or (j/n) to normalize the position
    
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + (i / m) # Deletion cost increases with index
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + (j / n) # Insertion cost increases with index

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate current weight (average of the two positions)
            current_weight = (i / m + j / n) / 2
            
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = current_weight

            dp[i][j] = min(
                dp[i-1][j] + current_weight,   # Deletion
                dp[i][j-1] + current_weight,   # Insertion
                dp[i-1][j-1] + cost            # Substitution
            )


    return dp[m][n]

"""


def compute_trajectory_metrics_for_run(
    trajectory_payload: Dict[str, Any],
) -> Dict[str, Any]:
    entries = trajectory_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("trajectory.json missing 'entries' list")

    by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if not isinstance(e, dict):
            continue
        ex_id = e.get("example_id")
        if ex_id is None:
            continue
        by_example[str(ex_id)].append(e)

    per_example_id: Dict[str, Dict[str, Any]] = {}

    total_pairs = 0
    total_jsd = 0.0
    total_norm_edit = 0.0
    mode = "unweighted"
    for ex_id, runs in sorted(by_example.items(), key=lambda kv: kv[0]):
        sequences = [_action_sequence(r) for r in runs]
        probs = _seqs_to_prob_matrix(sequences)

        k = len(sequences)
        pairs = k * (k - 1) // 2
        if pairs == 0:
            per_example_id[ex_id] = {
                "K": k,
                "pairs": 0,
                "C_traj_d": None,
                "C_traj_s": None,
                "C_traj": None,
            }
            continue

        jsd_sum = 0.0
        norm_edit_sum = 0.0
        for i, j in combinations(range(k), 2):
            jsd_sum += _jsd(probs[i], probs[j])

            a, b = sequences[i], sequences[j]
            denom = _normalization_denominator(a, b, mode=mode)
            if denom == 0:
                norm = 0.0
            else:
                norm = _levenshtein_distance(a, b) / float(denom)
            norm_edit_sum += norm

        mean_jsd = jsd_sum / float(pairs)
        mean_norm_edit = norm_edit_sum / float(pairs)

        c_d = 1.0 - mean_jsd
        c_s = 1.0 - mean_norm_edit
        c = 0.5 * (c_d + c_s)

        c_d = round(c_d, 3)
        c_s = round(c_s, 3)
        c = round(c, 3)

        per_example_id[ex_id] = {
            "K": k,
            "pairs": pairs,
            "C_traj_d": c_d,
            "C_traj_s": c_s,
            "C_traj": c,
        }

        total_pairs += pairs
        total_jsd += jsd_sum
        total_norm_edit += norm_edit_sum

    if total_pairs == 0:
        overall_mean_jsd = None
        overall_mean_norm_edit = None
        c_traj_d = None
        c_traj_s = None
        c_traj = None
    else:
        overall_mean_jsd = total_jsd / float(total_pairs)
        overall_mean_norm_edit = total_norm_edit / float(total_pairs)
        c_traj_d = round(1.0 - overall_mean_jsd, 3)
        c_traj_s = round(1.0 - overall_mean_norm_edit, 3)
        c_traj = round(0.5 * (c_traj_d + c_traj_s), 3)

    metrics_payload: Dict[str, Any] = {
        "C_traj_d": c_traj_d,
        "C_traj_s": c_traj_s,
        "C_traj": c_traj,
        "num_tasks": len(per_example_id),
        "num_pairs": total_pairs,
        "per_example_id": per_example_id,
    }
    return metrics_payload


def write_trajectory_metrics_for_run(
    run_dir: Path,
    trajectory_json: Path,
    output_path: Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    traj = _read_json(trajectory_json)
    metrics_payload = compute_trajectory_metrics_for_run(traj)

    if not dry_run:
        _write_json(output_path, metrics_payload)

    return metrics_payload


def update_result_json_for_run(
    run_dir: Path,
    result_json: Path,
    trajectory_json: Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Optional: also embed metrics into result.json (can make files crowded).
    """
    result = _read_json(result_json)
    trajectory_metrics_json = run_dir / "trajectory_metrics.json"
    metrics_payload = (
        _read_json(trajectory_metrics_json)
        if trajectory_metrics_json.is_file()
        else None
    )

    if metrics_payload is None:
        traj = _read_json(trajectory_json)
        metrics_payload = compute_trajectory_metrics_for_run(traj)

    if not isinstance(result, dict):
        raise ValueError(f"result.json is not a JSON object: {result_json}")

    result["trajectory_metrics"] = metrics_payload
    if not dry_run:
        _write_json(result_json, result)
    return metrics_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute trajectory consistency metrics from trajectory.json."
    )
    ap.add_argument(
        "outputs_path",
        type=str,
        help="Run directory (contains result.json and trajectory.json)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not modify result.json files",
    )
    ap.add_argument(
        "--update-result-json",
        action="store_true",
        help="Also add metrics into result.json under key 'trajectory_metrics'",
    )
    ap.add_argument(
        "--output-file",
        type=str,
        default="trajectory_metrics.json",
        help="Name of output metrics file (default: trajectory_metrics.json)",
    )

    args = ap.parse_args(argv)

    run_dir = Path(args.outputs_path).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    trajectory_json = run_dir / "trajectory.json"
    if not trajectory_json.is_file():
        raise SystemExit(f"Missing trajectory.json in: {run_dir}")

    output_path = run_dir / args.output_file

    metrics = write_trajectory_metrics_for_run(
        run_dir,
        trajectory_json,
        output_path=output_path,
        dry_run=bool(args.dry_run),
    )

    if args.update_result_json:
        result_json = run_dir / "result.json"
        if not result_json.is_file():
            raise SystemExit(f"Missing result.json in: {run_dir}")

        update_result_json_for_run(
            run_dir,
            result_json,
            trajectory_json,
            dry_run=bool(args.dry_run),
        )

    c = metrics.get("C_traj")
    print(f"{run_dir}: C_traj={c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
