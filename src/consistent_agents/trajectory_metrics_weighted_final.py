#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import math
from scipy.spatial.distance import jensenshannon

CONSTANT_MULTIPLIER = 3.0
LEVENSHTEIN_MODES = ("unweighted", "linear", "constant", "exponential")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )




def _resolve_relative_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _trajectory_payload_has_inline_entries(payload: Dict[str, Any]) -> bool:
    return isinstance(payload.get("entries"), list)


def _task_id_from_entry_file(entry_file: str) -> str:
    stem = Path(entry_file).stem
    task_id = stem.split("__", 1)[0]
    return task_id.removesuffix("_perturbed")


def _example_group_id(entry: Dict[str, Any]) -> Optional[str]:
    entry_file = entry.get("__entry_file")
    if isinstance(entry_file, str) and entry_file:
        return _task_id_from_entry_file(entry_file)

    ex_id = entry.get("example_id")
    if ex_id is None:
        return None

    return str(ex_id).removesuffix("_perturbed")


def _load_entry_files_from_manifest(
    trajectory_json: Path,
    trajectory_payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Path]:
    """
    Supports trajectory.json files that are manifests instead of inline payloads.

    Manifest shape:
      {
        "entries_dir": "trajectories/",
        "entry_files": ["activity001__base.json", ...]
      }

    Each listed file is loaded and converted into the same in-memory shape used
    by the original inline trajectory format: {"entries": [...]}
    """
    entry_files = trajectory_payload.get("entry_files")
    if not isinstance(entry_files, list):
        raise ValueError(
            "trajectory.json missing 'entries' list or manifest 'entry_files' list"
        )

    base_dir = trajectory_json.parent
    entries_dir_value = trajectory_payload.get("entries_dir", "")
    if not isinstance(entries_dir_value, str):
        raise ValueError("trajectory.json manifest field 'entries_dir' must be a string")

    entries_dir = _resolve_relative_path(base_dir, entries_dir_value)
    entries: List[Dict[str, Any]] = []
    missing_files: List[Path] = []

    for entry_file in entry_files:
        if not isinstance(entry_file, str) or not entry_file:
            continue

        entry_path = _resolve_relative_path(entries_dir, entry_file)
        if not entry_path.is_file():
            missing_files.append(entry_path)
            continue

        entry_payload = _read_json(entry_path)
        if isinstance(entry_payload, dict):
            entry_payload["__entry_file"] = entry_file
            entry_payload["__entry_path"] = str(entry_path)
            entries.append(entry_payload)

    if missing_files:
        missing_preview = ", ".join(str(p) for p in missing_files[:5])
        suffix = "" if len(missing_files) <= 5 else f", ... +{len(missing_files) - 5} more"
        raise FileNotFoundError(f"Missing trajectory entry files: {missing_preview}{suffix}")

    return {"entries": entries}, entries_dir


def load_trajectory_payload(trajectory_json: Path) -> Dict[str, Any]:
    payload = _read_json(trajectory_json)
    if not isinstance(payload, dict):
        raise ValueError(f"trajectory.json is not a JSON object: {trajectory_json}")

    if _trajectory_payload_has_inline_entries(payload):
        return payload

    resolved_payload, _ = _load_entry_files_from_manifest(trajectory_json, payload)
    return resolved_payload


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


def find_trajectory_json_files(trajectories_root: Path) -> List[Path]:
    trajectory_files: List[Path] = []

    for dataset_dir in sorted(p for p in trajectories_root.iterdir() if p.is_dir()):
        for agent_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            trajectory_files.extend(sorted(agent_dir.rglob("trajectory.json")))

    return trajectory_files


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def build_trajectory_index_rows(
    trajectories_root: Path,
    *,
    metrics_filename: str = "trajectory_metrics.json",
    scores_filename: str = "trajectory_scores.json",
    dry_run: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for trajectory_json in find_trajectory_json_files(trajectories_root):
        relative_trajectory_json = trajectory_json.relative_to(trajectories_root)
        parts = relative_trajectory_json.parts

        if len(parts) < 3:
            continue

        trajectory_dir = trajectory_json.parent
        metrics = write_trajectory_metrics_for_run(
            trajectory_dir,
            trajectory_json,
            output_path=trajectory_dir / metrics_filename,
            dry_run=dry_run,
        )

        if not dry_run and scores_filename != metrics_filename:
            _write_json(trajectory_dir / scores_filename, metrics)

        rows.append(
            {
                "dataset": parts[0],
                "agent": parts[1],
                "trajectory_path": str(relative_trajectory_json.parent),
                "C_traj_d": _csv_value(metrics.get("C_traj_d")),
                "C_traj_s": _csv_value(metrics.get("C_traj_s")),
                "C_traj": _csv_value(metrics.get("C_traj")),
            }
        )

    return rows


def write_trajectory_index_csv(
    trajectories_root: Path,
    output_path: Path,
    *,
    dry_run: bool,
    metrics_filename: str = "trajectory_metrics.json",
    scores_filename: str = "trajectory_scores.json",
) -> List[Dict[str, str]]:
    rows = build_trajectory_index_rows(
        trajectories_root,
        metrics_filename=metrics_filename,
        scores_filename=scores_filename,
        dry_run=dry_run,
    )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "agent",
                    "trajectory_path",
                    "C_traj_d",
                    "C_traj_s",
                    "C_traj",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    return rows



def compute_weighted_levenshtein_scores_for_run(
    trajectory_payload: Dict[str, Any],
) -> Dict[str, Any]:
    entries = trajectory_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("trajectory.json missing 'entries' list")

    by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if not isinstance(e, dict):
            continue
        ex_id = _example_group_id(e)
        if ex_id is None:
            continue
        by_example[ex_id].append(e)

    per_example_id: Dict[str, Dict[str, Any]] = {}
    total_norm_edit_by_mode = {mode: 0.0 for mode in LEVENSHTEIN_MODES}
    total_pairs = 0

    for ex_id, runs in sorted(by_example.items(), key=lambda kv: kv[0]):
        sequences = [_action_sequence(r) for r in runs]
        k = len(sequences)
        pairs = k * (k - 1) // 2

        example_scores: Dict[str, Any] = {"K": k, "pairs": pairs}
        if pairs == 0:
            for mode in LEVENSHTEIN_MODES:
                example_scores[mode] = None
            per_example_id[ex_id] = example_scores
            continue

        norm_edit_sum_by_mode = {mode: 0.0 for mode in LEVENSHTEIN_MODES}
        for i, j in combinations(range(k), 2):
            a, b = sequences[i], sequences[j]
            for mode in LEVENSHTEIN_MODES:
                denom = _normalization_denominator(a, b, mode=mode)
                norm = (
                    0.0
                    if denom == 0
                    else _levenshtein_distance(a, b, mode=mode) / float(denom)
                )
                norm_edit_sum_by_mode[mode] += norm

        for mode in LEVENSHTEIN_MODES:
            score = 1.0 - (norm_edit_sum_by_mode[mode] / float(pairs))
            example_scores[mode] = round(score, 3)
            total_norm_edit_by_mode[mode] += norm_edit_sum_by_mode[mode]

        per_example_id[ex_id] = example_scores
        total_pairs += pairs

    scores: Dict[str, Any] = {}
    for mode in LEVENSHTEIN_MODES:
        scores[mode] = None if total_pairs == 0 else round(
            1.0 - (total_norm_edit_by_mode[mode] / float(total_pairs)), 3
        )

    scores["num_tasks"] = len(per_example_id)
    scores["num_pairs"] = total_pairs
    scores["per_example_id"] = per_example_id
    return scores


def write_weighted_levenshtein_for_run(
    trajectory_json: Path,
    output_path: Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    traj = load_trajectory_payload(trajectory_json)
    scores = compute_weighted_levenshtein_scores_for_run(traj)

    if not dry_run:
        _write_json(output_path, scores)

    return scores


def build_weighted_levenshtein_rows(
    trajectories_root: Path,
    *,
    weighted_scores_filename: str = "weighted_levenshtein.json",
    dry_run: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for trajectory_json in find_trajectory_json_files(trajectories_root):
        relative_trajectory_json = trajectory_json.relative_to(trajectories_root)
        parts = relative_trajectory_json.parts

        if len(parts) < 3:
            continue

        trajectory_dir = trajectory_json.parent
        scores = write_weighted_levenshtein_for_run(
            trajectory_json,
            output_path=trajectory_dir / weighted_scores_filename,
            dry_run=dry_run,
        )

        rows.append(
            {
                "dataset": parts[0],
                "agent": parts[1],
                "trajectory_path": str(relative_trajectory_json.parent),
                "unweighted": _csv_value(scores.get("unweighted")),
                "linear": _csv_value(scores.get("linear")),
                "constant": _csv_value(scores.get("constant")),
                "exponential": _csv_value(scores.get("exponential")),
            }
        )

    return rows


def write_weighted_levenshtein_csv(
    trajectories_root: Path,
    output_path: Path,
    *,
    dry_run: bool,
    weighted_scores_filename: str = "weighted_levenshtein.json",
) -> List[Dict[str, str]]:
    rows = build_weighted_levenshtein_rows(
        trajectories_root,
        weighted_scores_filename=weighted_scores_filename,
        dry_run=dry_run,
    )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "agent",
                    "trajectory_path",
                    "unweighted",
                    "linear",
                    "constant",
                    "exponential",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    return rows


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
        ex_id = _example_group_id(e)
        if ex_id is None:
            continue
        by_example[ex_id].append(e)

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
    traj = load_trajectory_payload(trajectory_json)
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
        traj = load_trajectory_payload(trajectory_json)
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
        default=None,
        help=(
            "Output file. Defaults to trajectory_metrics.json for a single run "
            "and trajectory_index.csv for a trajectories folder."
        ),
    )
    ap.add_argument(
        "--scores-file",
        type=str,
        default="trajectory_scores.json",
        help="Per-run scores JSON written inside each trajectory folder when scanning a trajectories root.",
    )
    ap.add_argument(
        "--weighted-csv-file",
        type=str,
        default="weighted_levenshtein.csv",
        help="Separate CSV for sequence-only Levenshtein scores across unweighted, linear, constant, and exponential modes.",
    )
    ap.add_argument(
        "--weighted-scores-file",
        type=str,
        default="weighted_levenshtein.json",
        help="Per-run weighted Levenshtein sequence scores JSON written inside each trajectory folder.",
    )

    args = ap.parse_args(argv)

    run_dir = Path(args.outputs_path).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    trajectory_json = run_dir / "trajectory.json"
    if not trajectory_json.is_file():
        output_file = args.output_file or "trajectory_index.csv"
        output_path = run_dir / output_file
        rows = write_trajectory_index_csv(
            run_dir,
            output_path=output_path,
            dry_run=bool(args.dry_run),
            scores_filename=args.scores_file,
        )
        weighted_output_path = run_dir / args.weighted_csv_file
        write_weighted_levenshtein_csv(
            run_dir,
            output_path=weighted_output_path,
            dry_run=bool(args.dry_run),
            weighted_scores_filename=args.weighted_scores_file,
        )
        print(f"{run_dir}: found {len(rows)} trajectory.json files")
        if not args.dry_run:
            print(f"wrote CSV: {output_path}")
            print(f"wrote weighted Levenshtein CSV: {weighted_output_path}")
            print(f"wrote per-folder scores: {args.scores_file}")
            print(f"wrote per-folder weighted scores: {args.weighted_scores_file}")
        return 0

    output_file = args.output_file or "trajectory_metrics.json"
    output_path = run_dir / output_file

    metrics = write_trajectory_metrics_for_run(
        run_dir,
        trajectory_json,
        output_path=output_path,
        dry_run=bool(args.dry_run),
    )
    write_weighted_levenshtein_for_run(
        trajectory_json,
        output_path=run_dir / args.weighted_scores_file,
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
    if not args.dry_run:
        print(f"wrote weighted scores: {run_dir / args.weighted_scores_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
