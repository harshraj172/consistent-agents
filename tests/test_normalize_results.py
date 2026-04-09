"""Tests for scripts/normalize_results.py"""

import json
import pytest

from scripts.normalize_results import (
    is_success,
    extract_base_map,
    extract_pert_map,
    compute_consistency,
    aggregate_stats,
    normalize_all,
    _prompt_key,
)


class TestIsSuccess:
    def test_positive_float(self):
        assert is_success("1.0") is True

    def test_zero(self):
        assert is_success("0.0") is False

    def test_negative(self):
        assert is_success("-1.0") is False

    def test_empty(self):
        assert is_success("") is False

    def test_non_numeric(self):
        assert is_success("error") is False

    def test_none(self):
        assert is_success(None) is False


class TestExtractBaseMap:
    def test_swebench(self):
        data = {"examples": [
            {"id": "a", "base_output": "1.0", "perturbed_outputs": [{"output": "0.0"}]},
            {"id": "b", "base_output": "0.0", "perturbed_outputs": [{"output": "0.0"}]},
        ]}
        result = extract_base_map(data, is_spider2=False)
        assert result == {"a": "1.0", "b": "0.0"}

    def test_spider2_excludes_perturbed(self):
        data = {"examples": [
            {"id": "task001", "base_output": "1.0", "is_perturbed": False},
            {"id": "task001_perturbed", "base_output": "0.0", "is_perturbed": True},
        ]}
        result = extract_base_map(data, is_spider2=True)
        assert result == {"task001": "1.0"}
        assert "task001_perturbed" not in result

    def test_integer_ids(self):
        data = {"examples": [{"id": 123, "base_output": "1.0"}]}
        result = extract_base_map(data)
        assert "123" in result

    def test_key_by_prompt(self):
        data = {"examples": [
            {"id": "a", "base_output": "1.0", "base_prompt": "fix the bug in django"},
            {"id": "b", "base_output": "0.0", "base_prompt": "add feature to flask"},
        ]}
        result = extract_base_map(data, key_by="prompt")
        assert _prompt_key("fix the bug in django") in result
        assert result[_prompt_key("fix the bug in django")] == "1.0"


class TestExtractPertMap:
    def test_swebench_takes_first_pert(self):
        data = {"examples": [
            {"id": "a", "perturbed_outputs": [
                {"output": "1.0"}, {"output": "0.0"}, {"output": "1.0"}
            ]},
        ]}
        result = extract_pert_map(data, is_spider2=False)
        assert result == {"a": "1.0"}  # first pert only

    def test_swebench_no_perts(self):
        data = {"examples": [{"id": "a", "perturbed_outputs": []}]}
        result = extract_pert_map(data, is_spider2=False)
        assert result == {}

    def test_spider2(self):
        data = {"examples": [
            {"id": "task001", "base_output": "1.0", "is_perturbed": False},
            {"id": "task001_perturbed", "base_output": "0.0", "is_perturbed": True},
        ]}
        result = extract_pert_map(data, is_spider2=True)
        assert result == {"task001": "0.0"}


class TestComputeConsistency:
    def test_both_pass(self):
        base = {"a": "1.0"}
        pert = {"a": "1.0"}
        result = compute_consistency(base, pert)
        assert len(result) == 1
        assert result[0]["consistency"] == 1.0

    def test_both_fail(self):
        base = {"a": "0.0"}
        pert = {"a": "0.0"}
        result = compute_consistency(base, pert)
        assert result[0]["consistency"] == 1.0

    def test_disagree(self):
        base = {"a": "1.0"}
        pert = {"a": "0.0"}
        result = compute_consistency(base, pert)
        assert result[0]["consistency"] == 0.0

    def test_missing_pert_skipped(self):
        base = {"a": "1.0", "b": "0.0"}
        pert = {"a": "1.0"}  # b missing
        result = compute_consistency(base, pert)
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_extra_pert_ignored(self):
        base = {"a": "1.0"}
        pert = {"a": "1.0", "c": "0.0"}  # c not in base
        result = compute_consistency(base, pert)
        assert len(result) == 1

    def test_multiple_examples(self):
        base = {"a": "1.0", "b": "0.0", "c": "1.0"}
        pert = {"a": "1.0", "b": "1.0", "c": "0.0"}
        result = compute_consistency(base, pert)
        by_id = {r["id"]: r for r in result}
        assert by_id["a"]["consistency"] == 1.0  # both pass
        assert by_id["b"]["consistency"] == 0.0  # base fail, pert pass
        assert by_id["c"]["consistency"] == 0.0  # base pass, pert fail


class TestAggregateStats:
    def test_basic(self):
        per_example = [
            {"id": "a", "base_output": "1.0", "pert_output": "1.0", "base_pass": True, "pert_pass": True, "consistency": 1.0},
            {"id": "b", "base_output": "0.0", "pert_output": "1.0", "base_pass": False, "pert_pass": True, "consistency": 0.0},
        ]
        stats = aggregate_stats(per_example)
        assert stats["base_accuracy"] == 0.5
        assert stats["pert_accuracy"] == 1.0
        assert stats["consistency"] == 0.5
        assert stats["total"] == 2

    def test_all_consistent(self):
        per_example = [
            {"id": "a", "base_pass": True, "pert_pass": True, "consistency": 1.0, "base_output": "1.0", "pert_output": "1.0"},
            {"id": "b", "base_pass": False, "pert_pass": False, "consistency": 1.0, "base_output": "0.0", "pert_output": "0.0"},
        ]
        stats = aggregate_stats(per_example)
        assert stats["consistency"] == 1.0

    def test_empty(self):
        stats = aggregate_stats([])
        assert stats["total"] == 0
        assert stats["consistency"] is None


class TestNormalizeAllEndToEnd:
    def _make_swebench_result(self, examples):
        return {"config": {"n_perturbations": 1}, "status": "completed", "examples": examples}

    def _make_spider2_result(self, examples):
        return {"config": {"n_perturbations": 0}, "status": "completed", "examples": examples}

    def test_swebench_canonical_baseline(self, tmp_path):
        """All perturbation runs use the same canonical baseline for consistency."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        # noise run: base a=pass, b=fail
        noise_dir = src / "swebench" / "openhands-kimik2" / "noise"
        noise_dir.mkdir(parents=True)
        (noise_dir / "result.json").write_text(json.dumps(self._make_swebench_result([
            {"id": "a", "base_output": "1.0", "perturbed_outputs": [{"output": "1.0"}]},
            {"id": "b", "base_output": "0.0", "perturbed_outputs": [{"output": "0.0"}]},
        ])))

        # paraphrase run: base a=fail(!), b=pass(!) — different base results
        para_dir = src / "swebench" / "openhands-kimik2" / "paraphrase"
        para_dir.mkdir(parents=True)
        (para_dir / "result.json").write_text(json.dumps(self._make_swebench_result([
            {"id": "a", "base_output": "0.0", "perturbed_outputs": [{"output": "1.0"}]},
            {"id": "b", "base_output": "1.0", "perturbed_outputs": [{"output": "0.0"}]},
        ])))

        summary = normalize_all(str(src), str(dst))
        group = summary["swebench/openhands-kimik2"]

        # Canonical baseline is from noise: a=pass, b=fail
        assert group["base_accuracy"] == 0.5

        # Noise consistency: a base=pass pert=pass -> 1.0, b base=fail pert=fail -> 1.0
        assert group["runs"]["noise"]["consistency"] == 1.0

        # Paraphrase: compared against NOISE baseline (not its own base)
        # a: canonical=pass, pert=pass -> 1.0
        # b: canonical=fail, pert=fail -> 1.0
        assert group["runs"]["paraphrase"]["consistency"] == 1.0

        # Both report same base accuracy (from canonical)
        noise_out = json.load(open(dst / "swebench" / "openhands-kimik2" / "noise" / "result.json"))
        para_out = json.load(open(dst / "swebench" / "openhands-kimik2" / "paraphrase" / "result.json"))
        assert noise_out["base_accuracy"] == para_out["base_accuracy"]

    def test_swebench_3pert_truncated_to_1(self, tmp_path):
        """When example has 3 perts, only first is used for consistency."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        d = src / "swebench" / "openhands-kimik2" / "noise"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(self._make_swebench_result([
            {"id": "a", "base_output": "1.0", "perturbed_outputs": [
                {"output": "0.0"},  # first pert: fail
                {"output": "1.0"},  # second: pass (ignored)
                {"output": "1.0"},  # third: pass (ignored)
            ]},
        ])))

        summary = normalize_all(str(src), str(dst))
        # Consistency: base=pass, first_pert=fail -> 0.0
        assert summary["swebench/openhands-kimik2"]["runs"]["noise"]["consistency"] == 0.0

    def test_spider2_pairs_base_and_perturbed(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        # Baseline run
        bl_dir = src / "spider2-dbt" / "codex-gpt5mini" / "baseline"
        bl_dir.mkdir(parents=True)
        (bl_dir / "result.json").write_text(json.dumps(self._make_spider2_result([
            {"id": "task001", "base_output": "1.0", "is_perturbed": False},
            {"id": "task002", "base_output": "0.0", "is_perturbed": False},
        ])))

        # Timestamp perturbation run
        ts_dir = src / "spider2-dbt" / "codex-gpt5mini" / "timestamp"
        ts_dir.mkdir(parents=True)
        (ts_dir / "result.json").write_text(json.dumps(self._make_spider2_result([
            {"id": "task001", "base_output": "1.0", "is_perturbed": False},
            {"id": "task001_perturbed", "base_output": "0.0", "is_perturbed": True},
            {"id": "task002", "base_output": "0.0", "is_perturbed": False},
            {"id": "task002_perturbed", "base_output": "0.0", "is_perturbed": True},
        ])))

        summary = normalize_all(str(src), str(dst))
        group = summary["spider2-dbt/codex-gpt5mini"]

        # Canonical baseline from dedicated baseline run: task001=pass, task002=fail
        assert group["base_accuracy"] == 0.5

        # Timestamp: task001 canonical=pass pert=fail -> 0.0, task002 canonical=fail pert=fail -> 1.0
        assert group["runs"]["timestamp"]["consistency"] == 0.5

    def test_codex_gpt5mini_shared_baseline_via_prompt(self, tmp_path):
        """injection_swe and linear_mcp share baseline matched by prompt."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        # injection_swe: canonical baseline, uses instance IDs
        inj_dir = src / "swebench" / "codex-gpt5mini" / "injection_swe"
        inj_dir.mkdir(parents=True)
        (inj_dir / "result.json").write_text(json.dumps(self._make_swebench_result([
            {"id": "django-123", "base_output": "1.0", "base_prompt": "fix django bug",
             "perturbed_outputs": [{"output": "0.0"}]},
            {"id": "flask-456", "base_output": "0.0", "base_prompt": "add flask feature",
             "perturbed_outputs": [{"output": "0.0"}]},
        ])))

        # linear_mcp: different IDs but same prompts
        mcp_dir = src / "swebench" / "codex-gpt5mini" / "linear_mcp"
        mcp_dir.mkdir(parents=True)
        (mcp_dir / "result.json").write_text(json.dumps(self._make_swebench_result([
            {"id": 0, "base_output": "0.0", "base_prompt": "fix django bug",
             "perturbed_outputs": [{"output": "1.0"}]},
            {"id": 1, "base_output": "1.0", "base_prompt": "add flask feature",
             "perturbed_outputs": [{"output": "1.0"}]},
        ])))

        summary = normalize_all(str(src), str(dst))

        group = summary["swebench/codex-gpt5mini"]
        # Canonical base from injection_swe: django=pass, flask=fail -> 0.5
        assert group["base_accuracy"] == 0.5

        # injection_swe: django canonical=pass pert=fail->0, flask canonical=fail pert=fail->1
        assert group["runs"]["injection_swe"]["consistency"] == 0.5

        # linear_mcp: django canonical=pass pert=pass->1, flask canonical=fail pert=pass->0
        assert group["runs"]["linear_mcp"]["consistency"] == 0.5

    def test_output_files_written(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        d = src / "swebench" / "openhands-kimik2" / "noise"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(self._make_swebench_result([
            {"id": "a", "base_output": "1.0", "perturbed_outputs": [{"output": "1.0"}]},
        ])))

        normalize_all(str(src), str(dst))

        out_file = dst / "swebench" / "openhands-kimik2" / "noise" / "result.json"
        assert out_file.exists()

        out_data = json.load(open(out_file))
        assert out_data["canonical_baseline"] == "swebench/openhands-kimik2/noise"
        assert out_data["base_accuracy"] == 1.0
        assert len(out_data["examples"]) == 1
        assert out_data["examples"][0]["id"] == "a"
        assert out_data["examples"][0]["consistency"] == 1.0
