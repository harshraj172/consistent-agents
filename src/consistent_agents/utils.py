from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple


def resolve_object(dotted: str) -> Any:
    mod_path, _, attr = dotted.partition(":")
    if not attr:
        raise ValueError("Expected dotted path in format 'module.sub:attr'")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def maybe_instantiate(obj: Any, params: Optional[Dict[str, Any]] = None) -> Any:
    if isinstance(obj, type):
        return obj(**(params or {}))
    return obj


def _normalize_text(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

def compute_consistency(base: str, perturbed: List[str]) -> Tuple[bool, List[bool]]:
    base_n = _normalize_text(base)
    flags = [(_normalize_text(p) == base_n) for p in perturbed]
    return all(flags), flags