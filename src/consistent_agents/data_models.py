from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from consistent_agents.environments import BaseEnvironment


@dataclass
class BenchmarkItem:
    id: str | int
    prompt: str
    env: BaseEnvironment
    label: Optional[str] = None
    task_dir: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    instance_id: Optional[str] = None
    base_commit: Optional[str] = None
    repo: Optional[str] = None


@dataclass
class EvalConfig:
    n_perturbations: int = 5
    seed: int = 42


@dataclass
class ExampleResult:
    id: str
    base_output: str
    base_prompt: str
    perturbed_outputs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    consistency: Optional[List[Dict[str, Any]]] = None
    accuracy: Optional[float] = None



@dataclass
class AgentRunResult:
    output: str
    status: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    example_id: str
    variant: str
    prompt: str
    output: str
    status: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    config: Dict[str, Any]
    examples: List[ExampleResult]
    trajectories: List[AgentTrajectory] = field(default_factory=list)
    total: Optional[int] = None
    consistency: Optional[List[Dict[str, Any]]] = None
    accuracy: Optional[float] = None
