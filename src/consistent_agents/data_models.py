from __future__ import annotations

from dataclasses import dataclass
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
    perturbed_outputs: List[Dict[str, Any]]
    consistency: float
    accuracy: float
    bertscore: float
    rouge: float
    entailment: float
    contradiction: float

@dataclass
class EvalResult:
    config: Dict[str, Any]
    total: int
    consistency: float
    accuracy: float
    bertscore: float
    rouge: float
    entailment: float
    contradiction: float
    examples: List[ExampleResult]