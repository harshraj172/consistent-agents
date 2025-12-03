from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from consistent_agents.environments import BaseEnvironment


@dataclass
class BenchmarkItem:
    id: str
    prompt: str
    env: BaseEnvironment
    label: Optional[str] = None


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
