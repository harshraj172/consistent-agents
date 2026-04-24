#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon


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


def _levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """
    Levenshtein distance for sequences, O(len(a)*len(b)) time, O(min) memory.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is the shorter one for memory efficiency
    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


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
            denom = max(len(a), len(b))
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
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    traj = _read_json(trajectory_json)
    metrics_payload = compute_trajectory_metrics_for_run(traj)

    if not dry_run:
        _write_json(run_dir / "trajectory_metrics.json", metrics_payload)
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
    args = ap.parse_args(argv)

    run_dir = Path(args.outputs_path).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    trajectory_json = run_dir / "trajectory.json"
    if not trajectory_json.is_file():
        raise SystemExit(f"Missing trajectory.json in: {run_dir}")

    metrics = write_trajectory_metrics_for_run(
        run_dir, trajectory_json, dry_run=bool(args.dry_run)
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
