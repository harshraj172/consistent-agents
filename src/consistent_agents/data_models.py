from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkItem:
    id: str
    prompt: str
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
    consistency: int
    accuracy: int
    

@dataclass
class EvalResult:
    config: Dict[str, Any]
    total: int
    consistency: int
    accuracy: float
    examples: List[ExampleResult]
