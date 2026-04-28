#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import trajectory_metrics as tm


OUTPUT_FILES = {
    "unweighted": "trajectory_metrics_unweighted.json",
    "linear": "trajectory_metrics_weighted_linear.json",
    "constant": "trajectory_metrics_weighted_constant.json",
    "exponential": "trajectory_metrics_weighted_exponential.json",
}


def _sequence_distance(
    a: Sequence[str],
    b: Sequence[str],
    *,
    mode: str,
) -> float:
    if mode == "unweighted":
        return float(tm._levenshtein_distance(a, b, weighted=False))

    return float(tm._weighted_levenshtein(a, b, mode=mode))


def compute_trajectory_metrics_for_run_with_mode(
    trajectory_payload: Dict[str, Any],
    *,
    mode: str,
) -> Dict[str, Any]:
    entries = trajectory_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("trajectory.json missing 'entries' list")

    by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        example_id = entry.get("example_id")
        if example_id is None:
            continue

        by_example[str(example_id)].append(entry)

    per_example_id: Dict[str, Dict[str, Any]] = {}

    total_pairs = 0
    total_jsd = 0.0
    total_norm_edit = 0.0

    for example_id, runs in sorted(by_example.items(), key=lambda kv: kv[0]):
        sequences = [tm._action_sequence(run) for run in runs]
        probs = tm._seqs_to_prob_matrix(sequences)

        k = len(sequences)
        pairs = k * (k - 1) // 2

        if pairs == 0:
            per_example_id[example_id] = {
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
            jsd_sum += tm._jsd(probs[i], probs[j])

            a, b = sequences[i], sequences[j]
            denom = tm._normalization_denominator(a, b, mode=mode)

            if denom == 0:
                norm_edit = 0.0
            else:
                norm_edit = _sequence_distance(a, b, mode=mode) / float(denom)

            norm_edit_sum += norm_edit

        mean_jsd = jsd_sum / float(pairs)
        mean_norm_edit = norm_edit_sum / float(pairs)

        c_traj_d = round(1.0 - mean_jsd, 3)
        c_traj_s = round(1.0 - mean_norm_edit, 3)
        c_traj = round(0.5 * (c_traj_d + c_traj_s), 3)

        per_example_id[example_id] = {
            "K": k,
            "pairs": pairs,
            "C_traj_d": c_traj_d,
            "C_traj_s": c_traj_s,
            "C_traj": c_traj,
        }

        total_pairs += pairs
        total_jsd += jsd_sum
        total_norm_edit += norm_edit_sum

    if total_pairs == 0:
        c_traj_d = None
        c_traj_s = None
        c_traj = None
    else:
        overall_mean_jsd = total_jsd / float(total_pairs)
        overall_mean_norm_edit = total_norm_edit / float(total_pairs)

        c_traj_d = round(1.0 - overall_mean_jsd, 3)
        c_traj_s = round(1.0 - overall_mean_norm_edit, 3)
        c_traj = round(0.5 * (c_traj_d + c_traj_s), 3)

    return {
        "mode": mode,
        "C_traj_d": c_traj_d,
        "C_traj_s": c_traj_s,
        "C_traj": c_traj,
        "num_tasks": len(per_example_id),
        "num_pairs": total_pairs,
        "per_example_id": per_example_id,
    }


def find_run_dirs(root: Path, *, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(
            path.parent for path in root.rglob("trajectory.json") if path.is_file()
        )

    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "trajectory.json").is_file()
    )


def compute_all_modes_for_run(
    run_dir: Path,
    *,
    dry_run: bool,
) -> Dict[str, Dict[str, Any]]:
    trajectory_json = run_dir / "trajectory.json"
    trajectory_payload = tm._read_json(trajectory_json)

    metrics_by_mode: Dict[str, Dict[str, Any]] = {}

    for mode, output_file in OUTPUT_FILES.items():
        metrics = compute_trajectory_metrics_for_run_with_mode(
            trajectory_payload,
            mode=mode,
        )
        metrics_by_mode[mode] = metrics

        if not dry_run:
            tm._write_json(run_dir / output_file, metrics)

    return metrics_by_mode


def write_csv(
    csv_path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    fieldnames = [
        "folder_name",
        "run_dir",
        "unweighted_sequence_score",
        "weighted_linear_sequence_score",
        "weighted_constant_sequence_score",
        "weighted_exponential_sequence_score",
    ]

    if dry_run:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-compute unweighted and weighted trajectory sequence scores "
            "for every child run folder containing trajectory.json."
        )
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Parent folder containing run folders.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="trajectory_sequence_scores.csv",
        help="CSV filename or path. Default: trajectory_sequence_scores.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find trajectory.json files recursively instead of only immediate child folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print, but do not write JSON or CSV files.",
    )

    args = parser.parse_args(argv)

    root = Path(args.folder).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    csv_path = Path(args.csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    run_dirs = find_run_dirs(root, recursive=bool(args.recursive))
    if not run_dirs:
        raise SystemExit(f"No trajectory.json files found under: {root}")

    rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        metrics_by_mode = compute_all_modes_for_run(
            run_dir,
            dry_run=bool(args.dry_run),
        )

        row = {
            "folder_name": run_dir.name,
            "run_dir": str(run_dir),
            "unweighted_sequence_score": metrics_by_mode["unweighted"]["C_traj_s"],
            "weighted_linear_sequence_score": metrics_by_mode["weighted_linear"][
                "C_traj_s"
            ],
            "weighted_constant_sequence_score": metrics_by_mode["weighted_constant"][
                "C_traj_s"
            ],
            "weighted_exponential_sequence_score": metrics_by_mode[
                "weighted_exponential"
            ]["C_traj_s"],
        }
        rows.append(row)

        print(
            f"{run_dir.name}: "
            f"unweighted={row['unweighted_sequence_score']} "
            f"linear={row['weighted_linear_sequence_score']} "
            f"constant={row['weighted_constant_sequence_score']} "
            f"exponential={row['weighted_exponential_sequence_score']}"
        )

    write_csv(csv_path, rows, dry_run=bool(args.dry_run))

    if not args.dry_run:
        print(f"Wrote CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
