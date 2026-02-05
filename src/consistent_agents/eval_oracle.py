from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm.auto import tqdm


@dataclass
class OracleResult:
    instance_id: str
    perturbation: str
    passed: bool
    reward: Optional[float] = None
    error: Optional[str] = None
    task_dir: Optional[str] = None


def _slugify(text: str) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        else:
            safe.append("-")
    return "".join(safe).strip("-") or "run"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_object(path: str):
    if ":" in path:
        module_path, obj_name = path.rsplit(":", 1)
    else:
        module_path, obj_name = path.rsplit(".", 1)
    
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def instantiate_benchmark(bm_cfg: Dict[str, Any]):
    """Create benchmark instance from config."""
    path = bm_cfg.get("path")
    params = bm_cfg.get("params", {})
    
    cls = resolve_object(path)
    benchmark = cls(**params)
    
    if hasattr(benchmark, "load"):
        benchmark.load()
    
    return benchmark


def instantiate_perturbations(pert_cfgs: List[Dict[str, Any]]) -> List[Tuple[Any, Dict[str, Any]]]:
    """Create perturbation instances from config."""
    perturbations = []
    
    for cfg in pert_cfgs:
        path = cfg.get("path")
        params = cfg.get("params", {})
        
        cls = resolve_object(path)
        inst = cls(**params)
        perturbations.append((inst, cfg))
    
    return perturbations


def prepare_task_dir(
    source_dir: Path,
    workdir: Path,
    instance_id: str,
    pert_name: str,
    perturbation: Optional[Any] = None,
    pert_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Copy task directory and apply perturbation."""
    safe_name = _slugify(f"{instance_id}-oracle-{pert_name}")
    run_dir = workdir / f"{safe_name}-{uuid.uuid4().hex[:8]}"
    
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(source_dir, run_dir)
    
    # Apply task_dir perturbation if applicable
    if perturbation and getattr(perturbation, 'modifies_task_dir', False):
        if hasattr(perturbation, 'apply_to_task_dir'):
            perturbation.apply_to_task_dir(run_dir, **(pert_kwargs or {}))
    
    return run_dir


def run_oracle_with_harbor(
    task_dir: Path,
    harbor_cfg: Dict[str, Any],
    trials_dir: Path,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Run oracle validation using Harbor.
    
    Returns: (passed, reward, error)
    """
    try:
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            VerifierConfig,
            TrialConfig,
        )
        from harbor.trial.trial import Trial
    except ImportError as e:
        return False, None, f"Harbor not installed: {e}"
    
    trial_name = f"oracle-{uuid.uuid4().hex[:8]}"
    
    # Create configs
    task_cfg = TaskConfig(path=task_dir)
    
    # Use script agent to run solve.sh
    agent_cfg = AgentConfig(
        name="oracle",
        import_path="harbor.agents.script:ScriptAgent",
        kwargs={"script_path": str(task_dir / "solution" / "solve.sh")},
    )
    
    env_cfg_dict = harbor_cfg.get("environment", {})
    env_type = env_cfg_dict.pop("type", "docker")
    env_cfg = EnvironmentConfig(type=EnvironmentType(env_type), **env_cfg_dict)
    
    verifier_cfg = VerifierConfig(
        override_timeout_sec=harbor_cfg.get("verifier", {}).get("timeout_sec", 3000),
    )
    
    trial_config = TrialConfig(
        task=task_cfg,
        trial_name=trial_name,
        trials_dir=trials_dir,
        timeout_multiplier=harbor_cfg.get("timeout_multiplier", 1.0),
        agent=agent_cfg,
        environment=env_cfg,
        verifier=verifier_cfg,
    )
    
    async def run_async():
        trial = Trial(trial_config)
        return await trial.run()
    
    try:
        result = asyncio.run(run_async())
    except Exception as e:
        return False, None, f"Trial failed: {e}"
    
    if result.exception_info:
        return False, None, f"Exception: {result.exception_info.exception_type}"
    
    if result.verifier_result and result.verifier_result.rewards:
        reward_key = harbor_cfg.get("reward_key", "reward")
        rewards = result.verifier_result.rewards
        
        reward = rewards.get(reward_key) or next(iter(rewards.values()), None)
        if reward is not None:
            try:
                reward_float = float(reward)
                return reward_float > 0, reward_float, None
            except (TypeError, ValueError):
                pass
    
    return False, None, "No reward returned"


def run_oracle_simple(task_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Run oracle validation using simple Docker execution.
    
    This is a fallback when Harbor imports fail.
    Returns: (passed, error)
    """
    import subprocess
    
    # Build and run Docker container
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"
    
    if not dockerfile.exists():
        return False, "Dockerfile not found"
    
    image_name = f"oracle-test-{uuid.uuid4().hex[:8]}"
    
    try:
        # Build image
        build_result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", str(dockerfile), str(env_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if build_result.returncode != 0:
            return False, f"Docker build failed: {build_result.stderr[:500]}"
        
        # Run solve.sh
        solve_sh = task_dir / "solution" / "solve.sh"
        if not solve_sh.exists():
            return False, "solve.sh not found"
        
        # Copy solve.sh content and run
        solve_content = solve_sh.read_text()
        
        run_result = subprocess.run(
            ["docker", "run", "--rm", image_name, "bash", "-c", solve_content],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        # Run tests
        test_sh = task_dir / "tests" / "test.sh"
        if test_sh.exists():
            test_result = subprocess.run(
                ["docker", "run", "--rm", image_name, "bash", str(test_sh)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            passed = test_result.returncode == 0 or "PASSED" in test_result.stdout
            return passed, None if passed else f"Tests failed: {test_result.stdout[-500:]}"
        
        return run_result.returncode == 0, None
        
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        # Cleanup
        subprocess.run(["docker", "rmi", "-f", image_name], capture_output=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate oracle patches with perturbations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python run_oracle_validation.py config.yaml
            python run_oracle_validation.py config.yaml --keep-dirs
            python run_oracle_validation.py config.yaml --simple  # Skip Harbor, use Docker directly
                """,
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--keep-dirs", action="store_true", help="Keep task directories after validation")
    parser.add_argument("--simple", action="store_true", help="Use simple Docker execution instead of Harbor")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args(argv)
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    print(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    
    # Load benchmark
    print("Loading benchmark...")
    bm_cfg = cfg.get("benchmark", {})
    try:
        benchmark = instantiate_benchmark(bm_cfg)
    except Exception as e:
        print(f"Error loading benchmark: {e}")
        return 1
    
    # Collect items
    items = []
    for idx, ex in enumerate(benchmark):
        items.append({
            "id": idx,
            "instance_id": ex.get("instance_id", str(idx)),
            "prompt": ex.get("prompt", ""),
            "task_dir": ex.get("task_dir"),
            "repo": ex.get("repo"),
            "base_commit": ex.get("base_commit"),
        })
    
    print(f"Loaded {len(items)} items from benchmark")
    
    # Load perturbations
    pert_cfgs = cfg.get("perturbations", [])
    if isinstance(pert_cfgs, dict):
        pert_cfgs = [pert_cfgs]
    
    print(f"Loading {len(pert_cfgs)} perturbation(s)...")
    try:
        perturbations = instantiate_perturbations(pert_cfgs)
    except Exception as e:
        print(f"Error loading perturbations: {e}")
        return 1
    
    # Setup
    harbor_cfg = cfg.get("harbor", {})
    workdir = Path(harbor_cfg.get("task", {}).get("workdir", ".harbor_runs")) / "oracle_validation"
    workdir.mkdir(parents=True, exist_ok=True)
    
    trials_dir = Path(harbor_cfg.get("trials_dir", "harbor-trials"))
    trials_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[OracleResult] = []
    
    # Run validation
    print("\n" + "=" * 60)
    print("Running Oracle Validation")
    print("=" * 60)
    
    for item in tqdm(items, desc="Validating"):
        instance_id = item["instance_id"]
        task_dir = item.get("task_dir")
        
        if not task_dir or not Path(task_dir).exists():
            results.append(OracleResult(
                instance_id=instance_id,
                perturbation="base",
                passed=False,
                error="Task directory not found",
            ))
            continue
        
        task_dir = Path(task_dir)
        
        # Test base case
        base_run_dir = prepare_task_dir(task_dir, workdir, instance_id, "base")
        
        try:
            if args.simple:
                passed, error = run_oracle_simple(base_run_dir)
                reward = 1.0 if passed else 0.0
            else:
                passed, reward, error = run_oracle_with_harbor(base_run_dir, harbor_cfg, trials_dir)
            
            results.append(OracleResult(
                instance_id=instance_id,
                perturbation="base",
                passed=passed,
                reward=reward,
                error=error,
                task_dir=str(base_run_dir) if args.keep_dirs else None,
            ))
            
            if args.verbose:
                status = "✓" if passed else "✗"
                print(f"  {status} {instance_id} (base): reward={reward}")
                
        finally:
            if not args.keep_dirs:
                shutil.rmtree(base_run_dir, ignore_errors=True)
        
        # Test each perturbation
        for pert_inst, pert_cfg in perturbations:
            pert_name = pert_cfg.get("name", pert_inst.__class__.__name__)
            
            pert_kwargs = {
                "instance_id": instance_id,
                "problem_statement": item.get("prompt", ""),
                "repo": item.get("repo"),
                "base_commit": item.get("base_commit"),
            }
            
            pert_run_dir = prepare_task_dir(
                task_dir, workdir, instance_id, pert_name,
                perturbation=pert_inst,
                pert_kwargs=pert_kwargs,
            )
            
            try:
                if args.simple:
                    passed, error = run_oracle_simple(pert_run_dir)
                    reward = 1.0 if passed else 0.0
                else:
                    passed, reward, error = run_oracle_with_harbor(pert_run_dir, harbor_cfg, trials_dir)
                
                results.append(OracleResult(
                    instance_id=instance_id,
                    perturbation=pert_name,
                    passed=passed,
                    reward=reward,
                    error=error,
                    task_dir=str(pert_run_dir) if args.keep_dirs else None,
                ))
                
                if args.verbose:
                    status = "✓" if passed else "✗"
                    print(f"  {status} {instance_id} ({pert_name}): reward={reward}")
                    
            finally:
                if not args.keep_dirs:
                    shutil.rmtree(pert_run_dir, ignore_errors=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    
    print(f"\nTotal:  {total}")
    print(f"Passed: {passed_count} ({100*passed_count/total:.1f}%)" if total > 0 else "Passed: 0")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed cases:")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.instance_id} + {r.perturbation}: {r.error or 'unknown error'}")
    
    # Save results
    output_path = args.output or Path(cfg.get("output", {}).get("path", "outputs")).parent / "oracle_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": str(args.config),
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": passed_count / total if total > 0 else 0.0,
        "results": [asdict(r) for r in results],
    }
    
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\nResults saved to {output_path}")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())