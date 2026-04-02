from __future__ import annotations

import asyncio
import json
import random
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from tqdm.auto import tqdm

from consistent_agents.benchmarks import BaseBenchmark
from consistent_agents.data_models import (
    AgentRunResult,
    AgentTrajectory,
    BenchmarkItem,
    EvalConfig,
    EvalResult,
    ExampleResult,
)
from consistent_agents.utils import maybe_instantiate, resolve_object

# Harbor imports
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    AgentConfig as HarborAgentConfig,
    EnvironmentConfig as HarborEnvironmentConfig,
    TaskConfig as HarborTaskConfig,
    TrialConfig,
    VerifierConfig as HarborVerifierConfig,
)
from harbor.trial.trial import Trial


# ------------------------------------------------------------------------------
# Utility / Config helpers
# ------------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Return a Docker/compose-safe slug."""
    safe = []
    for ch in text.lower():
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        else:
            safe.append("-")
    slug = "".join(safe).strip("-")
    return slug or "run"


@dataclass
class HarborTaskOptions:
    template_path: Path
    instruction_file: str = "instruction.md"
    workdir: Path = Path(".harbor_tasks")
    keep: bool = False
    rewrite_instruction: bool = True


@dataclass
class HarborOptions:
    task: HarborTaskOptions
    agent: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    verifier: Dict[str, Any] = field(default_factory=dict)
    trials_dir: Path = Path("trials")
    timeout_multiplier: float = 1.0
    reward_key: Optional[str] = "reward"
    n_concurrent: int = 1  # Number of concurrent trials

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "HarborOptions":
        if "task" not in cfg or not cfg["task"].get("template_path"):
            raise ValueError("harbor.task.template_path must be provided")

        task_cfg = cfg["task"]
        task = HarborTaskOptions(
            template_path=Path(task_cfg["template_path"]),
            instruction_file=task_cfg.get("instruction_file", "instruction.md"),
            workdir=Path(task_cfg.get("workdir", ".harbor_tasks")),
            keep=bool(task_cfg.get("keep", False)),
            rewrite_instruction=bool(task_cfg.get("rewrite_instruction", True)),
        )

        return cls(
            task=task,
            agent=cfg.get("agent", {}),
            environment=cfg.get("environment", {}),
            verifier=cfg.get("verifier", {}),
            trials_dir=Path(cfg.get("trials_dir", "trials")),
            timeout_multiplier=float(cfg.get("timeout_multiplier", 1.0)),
            reward_key=cfg.get("reward_key", "reward"),
            n_concurrent=int(cfg.get("n_concurrent", 1)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": {
                "template_path": str(self.task.template_path),
                "instruction_file": self.task.instruction_file,
                "workdir": str(self.task.workdir),
                "keep": self.task.keep,
                "rewrite_instruction": self.task.rewrite_instruction,
            },
            "agent": self.agent,
            "environment": self.environment,
            "verifier": self.verifier,
            "trials_dir": str(self.trials_dir),
            "timeout_multiplier": self.timeout_multiplier,
            "reward_key": self.reward_key,
            "n_concurrent": self.n_concurrent,
        }


def load_benchmark_from_config(bm_cfg: Dict[str, Any]) -> Tuple[BaseBenchmark, List[BenchmarkItem]]:
    """Create a benchmark instance from config and return items."""
    path = bm_cfg.get("path")
    params = bm_cfg.get("params", {})
    if not path:
        raise ValueError("benchmark.path must be provided in config.yaml")

    obj = resolve_object(path)
    benchmark = maybe_instantiate(obj, params)
    if hasattr(benchmark, "load") and callable(getattr(benchmark, "load")):
        benchmark.load()

    items: List[BenchmarkItem] = []
    for ex_id, ex in enumerate(benchmark):
        prompt = ex["prompt"]
        if prompt is None:
            continue
        label = ex.get("label") if isinstance(ex.get("label"), (str, int)) else None
        task_dir = ex.get("task_dir")
        task_dir_path = Path(task_dir) if task_dir else None
        item_id = ex.get("instance_id", ex_id)
        metadata = dict(ex.get("metadata") or {})

        item = BenchmarkItem(
                id=str(item_id),
                prompt=str(prompt),
                label=str(label) if label is not None else None,
                env=ex["env"],
                task_dir=task_dir_path,
                metadata=metadata,
                base_commit=ex.get("base_commit"),
            )
        if "instance_id" in ex:
            item.instance_id = ex["instance_id"]
        if "base_commit" in ex:
            item.base_commit = ex.get("base_commit")
        if "repo" in ex:
            item.repo = ex.get("repo")
            
        items.append(item)
    return benchmark, items


def _instantiate_perturbations(cfg_list: List[Dict[str, Any]]) -> List[Tuple[Any, Dict[str, Any]]]:
    perts: List[Tuple[Any, Dict[str, Any]]] = []
    for pcfg in cfg_list:
        path = pcfg.get("path")
        params = pcfg.get("params", {})
        if not path:
            continue
        obj = resolve_object(path)
        inst = maybe_instantiate(obj, params)
        if not (hasattr(inst, "apply") and callable(getattr(inst, "apply"))) and not callable(inst):
            raise TypeError(f"Perturbation at {path} must be callable or implement .apply(text)")
        
        perts.append((inst, pcfg))
    return perts


def generate_perturbations(
    text: str,
    perturb_entries: List[Tuple[Any, Union[Dict[str, Any], str]]],
    n: int,
    seed: int,
    instance_id: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return list of (name, perturbed_text, instance) for the given input text."""
    rng = random.Random(seed)
    if not perturb_entries:
        return []
    perturb_entries = perturb_entries * n
    results: List[Tuple[str, str, Any]] = []
    for entry in perturb_entries:
        inst_or_fn, name_or_cfg = entry
        if isinstance(name_or_cfg, dict):
            name = name_or_cfg.get("name", getattr(inst_or_fn, "name", inst_or_fn.__class__.__name__))
        else:
            name = str(name_or_cfg)
        name_str = _slugify(str(name))
        if hasattr(inst_or_fn, "apply") and callable(getattr(inst_or_fn, "apply")):
            if getattr(inst_or_fn, "modifies_task_dir", False) and instance_id:
                perturbed = inst_or_fn.apply(text, instance_id=instance_id)
            else:
                perturbed = inst_or_fn.apply(text)
        elif callable(inst_or_fn):
            perturbed = inst_or_fn(text)
        else:
            perturbed = text
        results.append((name_str, perturbed, inst_or_fn))
    return results


# ------------------------------------------------------------------------------
# Harbor execution helpers
# ------------------------------------------------------------------------------


def _prepare_task_dir(
    prompt: str,
    harbor_cfg: HarborOptions,
    example_id: str,
    variant: str,
    source_task_dir: Optional[Path] = None,
    rewrite_instruction: bool = True,
    perturbation: Optional[Any] = None,
    perturbation_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Copy the template task and optionally inject the prompt into the instruction file.

    If `source_task_dir` is provided, that directory is copied verbatim; instruction
    rewriting can be disabled to preserve pre-generated Harbor tasks (e.g., SWEBench).
    """
    src = Path(source_task_dir) if source_task_dir is not None else harbor_cfg.task.template_path
    if not src.is_dir():
        raise FileNotFoundError(f"Harbor task template not found: {src}")

    harbor_cfg.task.workdir.mkdir(parents=True, exist_ok=True)
    safe_variant = _slugify(variant)
    run_dir = harbor_cfg.task.workdir / f"{example_id}-{safe_variant}-{uuid.uuid4().hex[:8]}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(src, run_dir)

    if perturbation is not None and getattr(perturbation, 'modifies_task_dir', False):
        kwargs = perturbation_kwargs or {}
        if hasattr(perturbation, 'apply_to_task_dir'):
            result = perturbation.apply_to_task_dir(run_dir, **kwargs)
            if getattr(perturbation, 'modifies_code', False):
                if hasattr(perturbation, '_transform_solution_dir'):
                    perturbation._transform_solution_dir(run_dir)
            if isinstance(result, dict) and result.get('prevent_instruction_rewrite'):
                rewrite_instruction = False
    if rewrite_instruction:
        instruction_path = run_dir / harbor_cfg.task.instruction_file
        instruction_path.parent.mkdir(parents=True, exist_ok=True)
        instruction_path.write_text(prompt, encoding="utf-8")
    return run_dir


def _parse_agent_config(cfg: Dict[str, Any]) -> HarborAgentConfig:
    agent_fields = {
        "name",
        "import_path",
        "model_name",
        "override_timeout_sec",
        "max_timeout_sec",
        "kwargs",
    }
    kwargs = {k: v for k, v in cfg.items() if k in agent_fields}
    return HarborAgentConfig(**kwargs)


def _parse_environment_config(cfg: Dict[str, Any]) -> HarborEnvironmentConfig:
    env_cfg = dict(cfg)
    env_type = env_cfg.pop("type", EnvironmentType.DOCKER.value)
    env_cfg["type"] = EnvironmentType(env_type)
    return HarborEnvironmentConfig(**env_cfg)


def _parse_verifier_config(cfg: Dict[str, Any]) -> HarborVerifierConfig:
    verifier_fields = {"override_timeout_sec", "max_timeout_sec", "disable"}
    kwargs = {k: v for k, v in cfg.items() if k in verifier_fields}
    return HarborVerifierConfig(**kwargs)


def _load_agent_messages(trials_dir: Path, trial_name: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Best-effort load of the Harbor agent trajectory for inclusion in the Eval trajectory.
    Returns the messages list (ATIF steps) and any metadata about the source or errors.
    """
    messages: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    traj_path = Path(trials_dir) / trial_name / "agent" / "trajectory.json"
    if not traj_path.is_file():
        return messages, meta

    meta["agent_trajectory_path"] = str(traj_path)

    try:
        payload = json.loads(traj_path.read_text())
        steps = payload.get("steps")
        if isinstance(steps, list):
            messages = steps
        else:
            meta["agent_trajectory_error"] = "trajectory.json missing 'steps' list"
    except Exception as exc:  # pragma: no cover - defensive parsing
        meta["agent_trajectory_error"] = f"failed to parse trajectory.json: {exc}"

    return messages, meta


def _extract_reward_text(rewards: Optional[Dict[str, Any]], reward_key: Optional[str]) -> str:
    if rewards is None:
        return ""
    if reward_key and reward_key in rewards:
        return str(rewards[reward_key])
    if rewards:
        first_val = next(iter(rewards.values()))
        return "" if first_val is None else str(first_val)
    return ""


async def run_harbor_trial_async(
    prompt: str,
    harbor_cfg: HarborOptions,
    example_id: str,
    variant: str,
    *,
    source_task_dir: Optional[Path] = None,
    rewrite_instruction: bool = True,
    perturbation: Optional[Any] = None,
    perturbation_kwargs: Optional[Dict[str, Any]] = None,
) -> AgentRunResult:
    """Run a single Harbor trial asynchronously and return an AgentRunResult with reward as output."""
    variant_slug = _slugify(str(variant))
    task_dir = _prepare_task_dir(
        prompt,
        harbor_cfg,
        str(example_id),
        variant_slug,
        source_task_dir=source_task_dir,
        rewrite_instruction=rewrite_instruction,
        perturbation=perturbation,
        perturbation_kwargs=perturbation_kwargs
    )
    trial_name = _slugify(f"{example_id}-{variant_slug}-{uuid.uuid4().hex[:8]}")

    task_cfg = HarborTaskConfig(path=task_dir)
    agent_cfg = _parse_agent_config(harbor_cfg.agent)
    env_cfg = _parse_environment_config(harbor_cfg.environment)
    verifier_cfg = _parse_verifier_config(harbor_cfg.verifier)

    trial_config = TrialConfig(
        task=task_cfg,
        trial_name=trial_name,
        trials_dir=harbor_cfg.trials_dir,
        timeout_multiplier=harbor_cfg.timeout_multiplier,
        agent=agent_cfg,
        environment=env_cfg,
        verifier=verifier_cfg,
    )

    metadata: Dict[str, Any] = {
        "trial_name": trial_name,
        "task_dir": str(task_dir),
        "trials_dir": str(harbor_cfg.trials_dir),
    }

    actual_instruction_path = task_dir / harbor_cfg.task.instruction_file
    if actual_instruction_path.is_file():
        metadata["instruction_prompt"] = actual_instruction_path.read_text(encoding="utf-8")

    try:
        trial = Trial(trial_config)
        trial_result = await trial.run()
    finally:
        if not harbor_cfg.task.keep:
            shutil.rmtree(task_dir, ignore_errors=True)

    status = "OK"
    reward_dict: Optional[Dict[str, Any]] = None

    if trial_result.exception_info is not None:
        status = trial_result.exception_info.exception_type
        metadata["exception"] = trial_result.exception_info.model_dump(mode="json")

    if trial_result.verifier_result is not None:
        reward_dict = trial_result.verifier_result.rewards
        metadata["reward"] = reward_dict

    output_text = _extract_reward_text(reward_dict, harbor_cfg.reward_key)
    messages, traj_meta = _load_agent_messages(harbor_cfg.trials_dir, trial_name)
    metadata.update(traj_meta)

    metadata["trial_result"] = trial_result.model_dump(mode="json", exclude_none=True)

    return AgentRunResult(
        output=output_text,
        status=status,
        messages=messages,
        metadata=metadata,
    )


def run_harbor_trial(
    prompt: str,
    harbor_cfg: HarborOptions,
    example_id: str,
    variant: str,
    *,
    source_task_dir: Optional[Path] = None,
    rewrite_instruction: bool = True,
) -> AgentRunResult:
    """Run a single Harbor trial synchronously (wrapper for async version)."""
    return asyncio.run(run_harbor_trial_async(
        prompt,
        harbor_cfg,
        example_id,
        variant,
        source_task_dir=source_task_dir,
        rewrite_instruction=rewrite_instruction,
    ))


# ------------------------------------------------------------------------------
# Evaluation loop with parallelism
# ------------------------------------------------------------------------------


async def _process_item_async(
    item: BenchmarkItem,
    harbor_cfg: HarborOptions,
    config: EvalConfig,
    perturb_fns: List[Tuple[Callable[[str], str], Dict[str, Any]]],
    semaphore: asyncio.Semaphore,
) -> Tuple[ExampleResult, List[AgentTrajectory], str, List[str]]:
    """Process a single benchmark item with all its perturbations."""
    async with semaphore:
        # Run base trial
        base_result = await run_harbor_trial_async(
            item.prompt,
            harbor_cfg,
            item.id,
            "base",
            source_task_dir=item.task_dir,
            rewrite_instruction=harbor_cfg.task.rewrite_instruction,
        )
        base_output = base_result.output

        trajectories = [
            AgentTrajectory(
                example_id=item.id,
                variant="base",
                prompt=item.prompt,
                output=base_output,
                status=base_result.status,
                messages=base_result.messages,
                metadata={**item.metadata, **dict(base_result.metadata)},
            )
        ]

        perts = generate_perturbations(
            item.prompt,
            perturb_fns,
            n=config.n_perturbations,
            seed=config.seed,
            instance_id=getattr(item, 'instance_id', None),
        )
        perturbed_outputs: List[Dict[str, Any]] = []
        pert_output_strs: List[str] = []

        for p_type, p_text, p_inst in perts:
            perturbation_kwargs = None
            is_task_dir_perturbation = getattr(p_inst, 'modifies_task_dir', False)
            if is_task_dir_perturbation:
                perturbation_kwargs = {
                    "instance_id": getattr(item, 'instance_id', None),
                    "problem_statement": item.prompt,
                    "repo": getattr(item, 'repo', None),
                    "base_commit": getattr(item, 'base_commit', None),
                }
            pert_result = await run_harbor_trial_async(
                p_text,
                harbor_cfg,
                item.id,
                p_type,
                source_task_dir=item.task_dir,
                rewrite_instruction=harbor_cfg.task.rewrite_instruction,
                perturbation=p_inst if is_task_dir_perturbation else None,
                perturbation_kwargs=perturbation_kwargs,
            )
            pert_output = pert_result.output

            pert_metadata = dict(pert_result.metadata)
            pert_metadata.setdefault("perturbation", p_type)
            pert_metadata = {**item.metadata, **pert_metadata}

            if is_task_dir_perturbation:
                task_mod = getattr(p_inst, 'last_result', {})
                pert_metadata["task_dir_modification"] = task_mod
                pert_metadata["is_perturbed"] = True
                if task_mod.get("rename"):
                    old_name, new_name = task_mod["rename"]
                    pert_metadata["variable_renamed"] = {
                        "old_name": old_name,
                        "new_name": new_name,
                    }
                if task_mod.get("solution_files_transformed"):
                    pert_metadata["solution_files_transformed"] = task_mod[
                        "solution_files_transformed"
                    ]
                if task_mod.get("fetched_extensions"):
                    pert_metadata["fetched_extensions"] = task_mod[
                        "fetched_extensions"
                    ]
                
            if getattr(p_inst, 'modifies_code', False):
                    pert_metadata["code_modification"] = getattr(p_inst, 'last_result', {})

            trajectories.append(
                AgentTrajectory(
                    example_id=item.id,
                    variant="perturbation",
                    prompt=pert_metadata.get("instruction_prompt", p_text),
                    output=pert_output,
                    status=pert_result.status,
                    messages=pert_result.messages,
                    metadata=pert_metadata,
                )
            )

            perturbed_outputs.append(
                {
                    "type": p_type,
                    "prompt": pert_metadata.get("instruction_prompt", p_text),
                    "output": pert_output,
                }
            )
            pert_output_strs.append(pert_output)

        example_result = ExampleResult(
            id=item.id,
            base_prompt=item.prompt,
            base_output=base_output,
            perturbed_outputs=perturbed_outputs,
            metadata=dict(item.metadata),
            accuracy=None,
            consistency=None,
        )

        return example_result, trajectories, base_output, pert_output_strs


async def evaluate_async(
    items: List[BenchmarkItem],
    benchmark: BaseBenchmark,
    config: EvalConfig,
    perturb_fns: List[Tuple[Callable[[str], str], Dict[str, Any]]],
    harbor_cfg: HarborOptions,
    run_dir: Optional[Path] = None,
    timestamp: Optional[datetime] = None,
) -> EvalResult:
    """Evaluate all items with concurrent execution."""
    n_concurrent = harbor_cfg.n_concurrent
    semaphore = asyncio.Semaphore(n_concurrent)

    print(f"Running evaluation with n_concurrent={n_concurrent}")

    # Create tasks for all items
    tasks = [
        _process_item_async(item, harbor_cfg, config, perturb_fns, semaphore)
        for item in items
    ]

    examples: List[ExampleResult] = []
    all_trajectories: List[AgentTrajectory] = []

    # Process with progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating", unit="ex"):
        example_result, trajectories, base_output, pert_outputs = await coro

        # Score the item
        benchmark.score(example_result.id, base_output, pert_outputs)
        item_score = benchmark.item_score()

        # Update example with scores
        example_result = ExampleResult(
            id=example_result.id,
            base_prompt=example_result.base_prompt,
            base_output=example_result.base_output,
            perturbed_outputs=example_result.perturbed_outputs,
            metadata=dict(example_result.metadata),
            **item_score,
        )

        examples.append(example_result)
        all_trajectories.extend(trajectories)

        # Save incrementally
        if run_dir is not None:
            _save_incremental_results(
                run_dir=run_dir,
                config={**asdict(config), "harbor": harbor_cfg.to_dict()},
                examples=examples,
                trajectories=all_trajectories,
                benchmark=benchmark,
                timestamp=timestamp or datetime.now(),
            )

    total_score = benchmark.total_score()
    return EvalResult(
        config={**asdict(config), "harbor": harbor_cfg.to_dict()},
        **total_score,
        examples=examples,
        trajectories=all_trajectories,
    )


def _save_incremental_results(
    run_dir: Path,
    config: Dict[str, Any],
    examples: List[ExampleResult],
    trajectories: List[AgentTrajectory],
    benchmark: BaseBenchmark,
    timestamp: datetime,
) -> None:
    """Save current results incrementally to disk."""
    total_score = benchmark.total_score()

    payload = {
        "config": config,
        "status": "in_progress",
        "completed_examples": len(examples),
        **{k: v for k, v in {
            "total": total_score.get("total"),
            "consistency": total_score.get("consistency"),
            "accuracy": total_score.get("accuracy"),
        }.items() if v is not None},
        "examples": [
            {
                "id": ex.id,
                "base_prompt": ex.base_prompt,
                "base_output": ex.base_output,
                "perturbed_outputs": ex.perturbed_outputs,
                "is_perturbed": bool(ex.metadata.get("is_perturbed", False)) or bool(ex.perturbed_outputs),
                "metadata": ex.metadata,
                **{k: v for k, v in {
                    "consistency": ex.consistency,
                    "accuracy": ex.accuracy,
                }.items() if v is not None},
            }
            for ex in examples
        ],
        "result_trajectory": "trajectory.json",
    }

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    traj_dir = run_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    entry_files = []
    for entry in trajectories:
        entry_dict = asdict(entry)
        filename = f"{entry.example_id}__{entry.variant}.json"
        entry_files.append(filename)
        (traj_dir / filename).write_text(
            json.dumps(entry_dict, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    trajectory_index = {
        "created_at": timestamp.isoformat(),
        "status": "in_progress",
        "entries_dir": "trajectories/",
        "trajectory_count": len(trajectories),
        "entry_files": entry_files,
    }
    trajectory_path = run_dir / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(trajectory_index, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def evaluate(
    items: List[BenchmarkItem],
    benchmark: BaseBenchmark,
    config: EvalConfig,
    perturb_fns: List[Tuple[Callable[[str], str], Dict[str, Any]]],
    harbor_cfg: HarborOptions,
    run_dir: Optional[Path] = None,
    timestamp: Optional[datetime] = None,
) -> EvalResult:
    """Evaluate with parallelism support."""
    return asyncio.run(evaluate_async(
        items, benchmark, config, perturb_fns, harbor_cfg, run_dir, timestamp
    ))


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def _load_config(config_path: str | Path) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_completed_ids(resume_dir: Path) -> set:
    """Load IDs of already-completed examples from a previous run's result.json."""
    result_path = resume_dir / "result.json"
    if not result_path.exists():
        print(f"No result.json found in {resume_dir}, starting fresh")
        return set()
    with open(result_path, "r") as f:
        data = json.load(f)
    completed = set()
    for ex in data.get("examples", []):
        completed.add(ex["id"])
    print(f"Resuming: skipping {len(completed)} already-completed examples")
    return completed


def main(argv: Optional[List[str]] = None) -> int:
    # Parse --resume flag
    args = list(argv) if argv else sys.argv[1:]
    resume_dir = None
    if "--resume" in args:
        idx = args.index("--resume")
        if idx + 1 < len(args):
            resume_dir = Path(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        else:
            print("--resume requires a path to a previous run directory")
            return 1

    cfg_path = Path(args[0]) if args else Path("config.yaml")
    raw_cfg = _load_config(cfg_path)
    n_perturbations = int(raw_cfg.get("eval", {}).get("n_perturbations", 5))
    eval_cfg = EvalConfig(
        n_perturbations=n_perturbations,
        seed=int(raw_cfg.get("eval", {}).get("seed", 42)),
    )

    harbor_cfg = HarborOptions.from_dict(raw_cfg.get("harbor", {}))

    # Benchmark
    benchmark, items = load_benchmark_from_config(raw_cfg.get("benchmark", {}))

    # Perturbations
    perturb_cfgs = raw_cfg.get("perturbations", [])
    if isinstance(perturb_cfgs, dict):
        perturb_cfgs = [perturb_cfgs]
    perturb_entries_raw = _instantiate_perturbations(perturb_cfgs)
    perturb_entries: List[Tuple[Callable[[str], str], Dict[str, Any]]] = []
    for (inst, _pcfg), cfg in zip(perturb_entries_raw, perturb_cfgs):
        name = cfg.get("name") or cfg.get("path") or "perturbation"
        perturb_entries.append((inst, name))

    # Resume: load completed IDs and pre-populate examples
    completed_ids: set = set()
    resumed_examples: List[ExampleResult] = []
    if resume_dir is not None:
        completed_ids = _load_completed_ids(resume_dir)
        # Load previous examples to merge later
        result_path = resume_dir / "result.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                prev_data = json.load(f)
            for ex in prev_data.get("examples", []):
                resumed_examples.append(ExampleResult(
                    id=ex["id"],
                    base_prompt=ex.get("base_prompt", ""),
                    base_output=ex.get("base_output", ""),
                    perturbed_outputs=ex.get("perturbed_outputs", []),
                    metadata=ex.get("metadata", {}),
                    consistency=ex.get("consistency"),
                    accuracy=ex.get("accuracy"),
                ))
        # Filter out completed items
        items = [it for it in items if it.id not in completed_ids]
        print(f"Remaining items to evaluate: {len(items)}")

    # Setup output directory early for incremental saves
    out_path_str = raw_cfg.get("output", {}).get("path") or raw_cfg.get("out")
    run_dir = None
    timestamp = datetime.now()

    if resume_dir is not None:
        # Reuse the resume directory
        run_dir = resume_dir
        print(f"Resuming into: {run_dir}")
    elif out_path_str:
        base_out = Path(out_path_str)
        if base_out.suffix:
            base_dir = base_out.parent
            base_name = base_out.stem
        else:
            base_dir = base_out
            base_name = base_out.name

        if not base_name:
            base_name = "results"

        run_dir_name = f"{base_name}-{timestamp.strftime('%Y-%m-%d::%H-%M-%S')}"
        run_dir = base_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results incrementally to: {run_dir}")

    # Evaluate with incremental saving
    result = evaluate(items, benchmark, eval_cfg, perturb_entries, harbor_cfg, run_dir, timestamp)

    # Merge resumed examples with new results
    if resumed_examples:
        all_examples = resumed_examples + result.examples
        result = EvalResult(
            config=result.config,
            total=len(all_examples),
            accuracy=result.accuracy,
            consistency=result.consistency,
            examples=all_examples,
            trajectories=result.trajectories,
        )

    # Final save
    payload = {
        "config": result.config,
        "status": "completed",
        **{
            k: v
            for k, v in {
                "total": result.total,
                "consistency": result.consistency,
                "accuracy": result.accuracy,
            }.items()
            if v is not None
        },
        "examples": [
            {
                "id": ex.id,
                "base_prompt": ex.base_prompt,
                "base_output": ex.base_output,
                "perturbed_outputs": ex.perturbed_outputs,
                "is_perturbed": bool(ex.metadata.get("is_perturbed", False)) or bool(ex.perturbed_outputs),
                "metadata": ex.metadata,
                **{
                    k: v
                    for k, v in {
                        "consistency": ex.consistency,
                        "accuracy": ex.accuracy,
                    }.items()
                    if v is not None
                },
            }
            for ex in result.examples
        ],
    }

    if run_dir:
        trajectory_filename = "trajectory.json"
        trajectory_path = run_dir / trajectory_filename
        result_path = run_dir / "result.json"

        payload["result_trajectory"] = trajectory_filename

        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        traj_dir = run_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        entry_files = []
        for entry in result.trajectories:
            entry_dict = asdict(entry)
            filename = f"{entry.example_id}__{entry.variant}.json"
            entry_files.append(filename)
            (traj_dir / filename).write_text(
                json.dumps(entry_dict, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        trajectory_index = {
            "created_at": timestamp.isoformat(),
            "status": "completed",
            "results_dir": run_dir.name,
            "trajectory_file": trajectory_filename,
            "entries_dir": "trajectories/",
            "trajectory_count": len(result.trajectories),
            "entry_files": entry_files,
        }
        trajectory_path.write_text(
            json.dumps(trajectory_index, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print(f"Wrote final results to {result_path} and trajectory log to {trajectory_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
