from __future__ import annotations

import pytest

from consistent_agents.benchmarks.swebench.swebench_benchmark import SWEBenchBenchmark


def test_oracle_patch_passes() -> None:


    benchmark = SWEBenchBenchmark()
    benchmark.load()

    for example_item in benchmark:
        example_item = next(iter(benchmark))
        example_item["env"].expected_patch = example_item["label"]

        idx = benchmark.get_index(example_item["instance_id"])
        result = benchmark.score(idx, example_item["label"], [example_item["label"]])

        assert result["base_passed"] == pytest.approx(1.0)
        assert result["consistent_count"] == 1
        assert result["correct_count"] == 1
        assert result["total"] == 1
