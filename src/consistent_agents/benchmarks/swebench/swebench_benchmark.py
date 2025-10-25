from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import datasets
from datasets import Dataset

from consistent_agents.benchmarks.base import BaseBenchmark
from consistent_agents.environments import DockerEnvironment

from consistent_agents.benchmarks.swebench.utils import get_image_names, get_test_commands

START_MARKER = "SWEBench results starts here"
END_MARKER = "SWEBench results ends here"


class SWEBenchBenchmark(BaseBenchmark):
    """Prepare SWEBench tasks inside the provided environment."""

    def __init__(
        self,
        split: str = "test",
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    ) -> None:
        super().__init__(split=split)
        self.dataset_name = dataset_name
        self.dataset: Optional[Dataset] = None
        self._id_to_index: Dict[str, int] = {}
        self._prepared: Dict[int, Dict] = {}
        self.template_dir = Path(__file__).parent / "template"

    def load(self) -> None:
        """Load the SWEBench dataset from HuggingFace."""
        self.dataset = datasets.load_dataset(
            self.dataset_name,
            split=self.split,
        )
        self._id_to_index = {
            example["instance_id"]: idx for idx, example in enumerate(self.dataset)
        }
        self.id_to_docker_image = get_image_names(list(self.dataset))
        
    def _prepare_instance(self, idx: int, example: Dict[str, Any]) -> Dict[int, Dict]:
        """Clone the repository at the bugged commit inside the environment."""
        instance_id = example["instance_id"]
        out_root = Path(tempfile.mkdtemp(prefix="swebench_"))

        docker_image = self.id_to_docker_image[instance_id]

        dockerfile_template = (self.template_dir / "Dockerfile").read_text()
        dockerfile_path = out_root / "Dockerfile"
        dockerfile_path.write_text(dockerfile_template.replace("{docker_image}", docker_image))

        tests_template_dir = self.template_dir / "tests"
        tests_dir = out_root / "tests"
        shutil.copytree(tests_template_dir, tests_dir)

        (tests_dir / "config.json").write_text(json.dumps(example, indent=2))

        run_tests_path = tests_dir / "run-tests.sh"
        test_sh_template = run_tests_path.read_text()
        test_commands = get_test_commands(
            example["test_patch"], example["repo"], example["version"], example["base_commit"]
        )
        run_tests_path.write_text(test_sh_template.replace("{test_commands}", test_commands))
        run_tests_path.chmod(0o755)

        env = DockerEnvironment()
        env.stop()
        env.start(str(dockerfile_path), f"swebench-{instance_id}")

        state = {
            "instance_id": instance_id,
            "env": env,
            "tests_dir": tests_dir,
            "label": example["patch"]
        }
        self._prepared[idx] = state
        return state

    def _format_prompt(self, example: Dict[str, Any], state: Dict[int, Dict]) -> str:
        instruction = example["problem_statement"]
        return instruction

    def iter(self) -> Iterator[Dict[str, Any]]:
        """Yield dataset entries after preparing each instance inside the environment."""
        if self.dataset is None:
            self.load()

        for idx, example in enumerate(self.dataset):
            state = self._prepared.get(idx)
            if state is None:
                state = self._prepare_instance(idx, example)
            yield {
                "instance_id": state["instance_id"],
                "prompt": self._format_prompt(example, state),
                "env": state["env"],
                "label": state["label"]
            }

    @staticmethod
    def _parse_test_output(output: str) -> bool:
        """Return True if the SWEBench parser reported success."""
        if not output or START_MARKER not in output or END_MARKER not in output:
            return False
        segment = output.split(START_MARKER, 1)[1].split(END_MARKER, 1)[0]
        return "PASSED" in segment

    def _apply_patch_and_test(self, env, patch) -> bool:
        solution_template = (self.template_dir / "solution.sh").read_text()
        script_contents = solution_template.replace("{patch}", patch)

        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp_file:
            tmp_file.write(script_contents)
            tmp_path = Path(tmp_file.name)

        uploaded = False
        try:
            uploaded = env.upload(str(tmp_path), "/testbed/solution.sh")
            if not uploaded:
                return False

            apply_result = env.execute("bash solution.sh false", cwd="/testbed", timeout=300)
            applied = apply_result.get("returncode", -1) == 0
            if not applied:
                return False

            test_result = env.execute("bash /tests/run-tests.sh", cwd="/testbed", timeout=3600)
            passed = self._parse_test_output(test_result.get("stdout", ""))
            return passed
        finally:
            try:
                if uploaded:
                    env.execute("bash solution.sh true", cwd="/testbed", timeout=300)
            finally:
                tmp_path.unlink(missing_ok=True)
    
    def score(
        self,
        idx: int,
        base_output: str,
        predictions: List[Any], 
    ) -> Dict[str, float]:
        """Score predictions by applying patches and running the SWEBench harness."""
        state = self._prepared[idx]
        env = state["env"]

        env.upload(str(state["tests_dir"]), "/")

        base_passed = self._apply_patch_and_test(env, str(base_output))

        outcomes: List[bool] = []
        for prediction in predictions:
            pred_passed = self._apply_patch_and_test(env, str(prediction))
            outcomes.append(pred_passed)

        consistent_count = correct_count = sum(1 for passed in outcomes if passed)
        total = len(outcomes) if outcomes else 1
        
        # env.stop()
        
        return {
            "consistent_count": consistent_count,
            "correct_count": correct_count,
            "total": total,
            "base_passed": float(base_passed),
        }

    def __len__(self) -> int:
        """Return the number of examples in the benchmark."""
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        return len(self.dataset)

    def get_index(self, instance_id: str) -> Optional[int]:
        """Return the dataset index for a given instance_id."""
        if self.dataset is None:
            self.load()
        return self._id_to_index.get(instance_id)
