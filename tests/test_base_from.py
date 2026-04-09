"""Tests for the --base-from feature in eval_harbor.py."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import pytest

from consistent_agents.data_models import (
    AgentRunResult,
    AgentTrajectory,
    BenchmarkItem,
    EvalConfig,
    ExampleResult,
)
from consistent_agents.eval_harbor import (
    HarborOptions,
    HarborTaskOptions,
    _load_base_results,
    _process_item_async,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_result_json(examples: List[Dict[str, Any]]) -> str:
    """Create a minimal result.json payload."""
    return json.dumps({"examples": examples}, indent=2)


def _make_mock_item(item_id: str = "test-001", prompt: str = "solve it") -> BenchmarkItem:
    """Return a lightweight BenchmarkItem for testing."""
    return BenchmarkItem(
        id=item_id,
        prompt=prompt,
        env=None,  # type: ignore[arg-type]
        label=None,
        task_dir=None,
        metadata={"source": "test"},
    )


def _make_harbor_cfg() -> HarborOptions:
    """Return a minimal HarborOptions (won't actually run trials)."""
    return HarborOptions(
        task=HarborTaskOptions(template_path=Path("/tmp/fake_template")),
        agent={"name": "test", "import_path": "fake.Agent"},
        environment={"type": "docker"},
        verifier={},
    )


# ---------------------------------------------------------------------------
# Test _load_base_results
# ---------------------------------------------------------------------------


class TestLoadBaseResults:
    def test_parses_examples(self, tmp_path: Path):
        examples = [
            {"id": "ex-1", "base_output": "output-1"},
            {"id": "ex-2", "base_output": "output-2"},
            {"id": "ex-3", "base_output": ""},
        ]
        (tmp_path / "result.json").write_text(_make_result_json(examples))

        bases = _load_base_results(tmp_path)

        assert len(bases) == 3
        assert bases["ex-1"] == "output-1"
        assert bases["ex-2"] == "output-2"
        assert bases["ex-3"] == ""

    def test_missing_result_json_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _load_base_results(tmp_path)

    def test_missing_base_output_defaults_empty(self, tmp_path: Path):
        examples = [{"id": "no-output"}]
        (tmp_path / "result.json").write_text(_make_result_json(examples))

        bases = _load_base_results(tmp_path)
        assert bases["no-output"] == ""

    def test_integer_ids_converted_to_str(self, tmp_path: Path):
        examples = [{"id": 42, "base_output": "answer"}]
        (tmp_path / "result.json").write_text(_make_result_json(examples))

        bases = _load_base_results(tmp_path)
        assert bases["42"] == "answer"


# ---------------------------------------------------------------------------
# Test _process_item_async with precomputed_base
# ---------------------------------------------------------------------------


class TestProcessItemAsyncPrecomputed:
    """Verify that _process_item_async skips the base trial when precomputed_base is provided."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("consistent_agents.eval_harbor.run_harbor_trial_async")
    def test_skips_base_trial(self, mock_run: AsyncMock):
        """When precomputed_base is given, run_harbor_trial_async should only be called for perturbations."""
        # No perturbations => run_harbor_trial_async should NOT be called at all
        mock_run.return_value = AgentRunResult(
            output="should-not-be-called", status="OK", messages=[], metadata={}
        )

        item = _make_mock_item()
        harbor_cfg = _make_harbor_cfg()
        config = EvalConfig(n_perturbations=1, seed=42)
        semaphore = asyncio.Semaphore(1)

        result, trajectories, base_output, pert_outputs = self._run(
            _process_item_async(
                item, harbor_cfg, config, [], semaphore,
                precomputed_base="precomputed-value",
            )
        )

        # Base trial should NOT have called run_harbor_trial_async (no perturbations either)
        mock_run.assert_not_called()

        # Verify base output is the precomputed value
        assert base_output == "precomputed-value"
        assert result.base_output == "precomputed-value"

        # Verify trajectory
        assert len(trajectories) == 1
        assert trajectories[0].variant == "base"
        assert trajectories[0].status == "precomputed"
        assert trajectories[0].messages == []
        assert trajectories[0].output == "precomputed-value"

    @patch("consistent_agents.eval_harbor.run_harbor_trial_async")
    def test_runs_base_trial_when_no_precomputed(self, mock_run: AsyncMock):
        """When precomputed_base is None, run_harbor_trial_async should be called for the base trial."""
        mock_run.return_value = AgentRunResult(
            output="computed-base", status="OK", messages=[{"role": "assistant"}], metadata={"trial_name": "t1"}
        )

        item = _make_mock_item()
        harbor_cfg = _make_harbor_cfg()
        config = EvalConfig(n_perturbations=1, seed=42)
        semaphore = asyncio.Semaphore(1)

        result, trajectories, base_output, pert_outputs = self._run(
            _process_item_async(
                item, harbor_cfg, config, [], semaphore,
                precomputed_base=None,
            )
        )

        # Should have called run_harbor_trial_async once for the base trial
        mock_run.assert_called_once()
        assert base_output == "computed-base"
        assert trajectories[0].status == "OK"
        assert trajectories[0].messages == [{"role": "assistant"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
