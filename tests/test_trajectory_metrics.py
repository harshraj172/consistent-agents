from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from consistent_agents.trajectory_metrics import (
    _extract_actions_from_atif_steps,
    _seqs_to_prob_matrix,
    compute_trajectory_metrics_for_run,
    update_result_json_for_run,
    write_trajectory_metrics_for_run,
)


def test_extract_actions_from_atif_steps_filters_correctly() -> None:
    steps = [
        {"source": "user", "tool_calls": [{"function_name": "should_not_count"}]},
        {"source": "agent", "tool_calls": []},
        {"source": "agent", "tool_calls": "not-a-list"},
        {
            "source": "agent",
            "tool_calls": [
                {"function_name": "tool_a"},
                {"function_name": ""},
                {"function_name": None},
                {"wrong_key": "tool_x"},
                "not-a-dict",
                {"function_name": "tool_b"},
            ],
        },
        {"source": "agent"},
    ]

    actions = _extract_actions_from_atif_steps(steps)
    assert actions == ["tool_a", "tool_b"]


def test_seqs_to_prob_matrix_normalizes_counts_and_handles_empty_vocab() -> None:
    seqs = [["a", "a", "b"], ["b", "c"], []]
    mat = _seqs_to_prob_matrix(seqs)

    assert mat.shape == (3, 3)
    np.testing.assert_allclose(mat[0], [2.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(mat[1], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(mat[2], [0.0, 0.0, 0.0])

    empty_vocab = _seqs_to_prob_matrix([[], []])
    assert empty_vocab.shape == (2, 0)


def test_compute_trajectory_metrics_for_run_mixed_examples() -> None:
    payload = {
        "entries": [
            {
                "example_id": "A",
                "messages": [
                    {"source": "agent", "tool_calls": [{"function_name": "search"}]},
                    {"source": "agent", "tool_calls": [{"function_name": "open"}]},
                ],
            },
            {
                "example_id": "A",
                "messages": [
                    {"source": "agent", "tool_calls": [{"function_name": "search"}]},
                    {"source": "agent", "tool_calls": [{"function_name": "open"}]},
                ],
            },
            {
                "example_id": "B",
                "messages": [
                    {"source": "agent", "tool_calls": [{"function_name": "search"}]},
                    {"source": "agent", "tool_calls": [{"function_name": "open"}]},
                ],
            },
            {
                "example_id": "B",
                "messages": [
                    {"source": "assistant", "tool_calls": [{"function_name": "ignored"}]},
                ],
            },
        ]
    }

    metrics = compute_trajectory_metrics_for_run(payload)

    assert metrics["num_tasks"] == 2
    assert metrics["num_pairs"] == 2
    assert metrics["C_traj_d"] == pytest.approx(0.5)
    assert metrics["C_traj_s"] == pytest.approx(0.5)
    assert metrics["C_traj"] == pytest.approx(0.5)

    ex_a = metrics["per_example_id"]["A"]
    assert ex_a["K"] == 2
    assert ex_a["pairs"] == 1
    assert ex_a["C_traj_d"] == pytest.approx(1.0)
    assert ex_a["C_traj_s"] == pytest.approx(1.0)
    assert ex_a["C_traj"] == pytest.approx(1.0)

    ex_b = metrics["per_example_id"]["B"]
    assert ex_b["K"] == 2
    assert ex_b["pairs"] == 1
    assert ex_b["C_traj_d"] == pytest.approx(0.0)
    assert ex_b["C_traj_s"] == pytest.approx(0.0)
    assert ex_b["C_traj"] == pytest.approx(0.0)


def test_compute_trajectory_metrics_for_run_single_run_and_invalid_entries() -> None:
    payload_single = {
        "entries": [
            {
                "example_id": "only",
                "messages": [{"source": "agent", "tool_calls": [{"function_name": "search"}]}],
            },
            {"example_id": None, "messages": []},
            "not-a-dict-entry",
        ]
    }

    metrics = compute_trajectory_metrics_for_run(payload_single)
    assert metrics["num_tasks"] == 1
    assert metrics["num_pairs"] == 0
    ex = metrics["per_example_id"]["only"]
    assert ex["pairs"] == 0
    assert ex["C_traj_d"] is None
    assert ex["C_traj_s"] is None
    assert ex["C_traj"] is None
    assert metrics["C_traj_d"] is None
    assert metrics["C_traj_s"] is None
    assert metrics["C_traj"] is None

    with pytest.raises(ValueError, match="missing 'entries' list"):
        compute_trajectory_metrics_for_run({})

    with pytest.raises(ValueError, match="missing 'entries' list"):
        compute_trajectory_metrics_for_run({"entries": "not-a-list"})


def test_write_and_update_result_json_roundtrip(tmp_path) -> None:
    run_dir = tmp_path
    trajectory_json = run_dir / "trajectory.json"
    result_json = run_dir / "result.json"

    trajectory_payload = {
        "entries": [
            {
                "example_id": "A",
                "messages": [{"source": "agent", "tool_calls": [{"function_name": "search"}]}],
            },
            {
                "example_id": "A",
                "messages": [{"source": "agent", "tool_calls": [{"function_name": "search"}]}],
            },
        ]
    }
    trajectory_json.write_text(json.dumps(trajectory_payload), encoding="utf-8")
    result_json.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    written_metrics = write_trajectory_metrics_for_run(run_dir, trajectory_json, dry_run=False)
    metrics_path = run_dir / "trajectory_metrics.json"
    assert metrics_path.is_file()

    on_disk_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert on_disk_metrics == written_metrics

    updated_metrics = update_result_json_for_run(
        run_dir,
        result_json,
        trajectory_json,
        dry_run=False,
    )
    updated_result = json.loads(result_json.read_text(encoding="utf-8"))
    assert updated_result["trajectory_metrics"] == updated_metrics
    assert updated_result["trajectory_metrics"] == written_metrics

    dry_run_dir = tmp_path / "dry_run"
    dry_run_dir.mkdir()
    dry_traj = dry_run_dir / "trajectory.json"
    dry_traj.write_text(json.dumps(trajectory_payload), encoding="utf-8")

    write_trajectory_metrics_for_run(dry_run_dir, dry_traj, dry_run=True)
    assert not (dry_run_dir / "trajectory_metrics.json").exists()
