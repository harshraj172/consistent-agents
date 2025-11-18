from __future__ import annotations

import pytest

from consistent_agents.benchmarks.swebench.swebench_benchmark import SWEBenchBenchmark


def test_oracle_patch_passes() -> None:

    num_test_samples = 1
    benchmark = SWEBenchBenchmark()
    benchmark.load()

    for i, example_item in enumerate(benchmark):
        example_item = next(iter(benchmark))

        idx = benchmark.get_index(example_item["instance_id"])
        result = benchmark.score(idx, example_item["label"], [example_item["label"]])

        assert result["base_passed"] == pytest.approx(1.0)
        assert result["consistent_count"] == 2
        assert result["correct_count"] == 2
        assert result["total"] == 2

        if i==num_test_samples:
            break