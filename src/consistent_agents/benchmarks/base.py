from abc import ABC, abstractmethod
from typing import Iterator, Any, Dict, Union


class BaseBenchmark(ABC):
    """Base class for all benchmarks.

    This base class is intentionally lightweight to accommodate different
    benchmark shapes. Subclasses are expected to implement a `load()` method
    and either an `iter()` generator or `__iter__` to yield examples.
    """

    def __init__(self, split: str = "validation", **kwargs):
        self.split = split
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> None:
        """Load any required data/resources."""
        raise NotImplementedError

    # Subclasses can implement either `iter()` or `__iter__`.
    def iter(self) -> Iterator[Dict[str, Any]]:  # pragma: no cover - optional
        """Optional: yield examples as dictionaries."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict[str, Any]]:  # pragma: no cover - optional
        return self.iter()

    # Optional scoring contract. Concrete benchmarks may override the
    # signature and semantics as needed.
    def score(
        self,
        prediction: Union[str, int, Dict],
        ground_truth: Union[str, int, Dict],
        example_id: str | None = None,
    ) -> Dict[str, float]:  # pragma: no cover - optional
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - optional
        """Best-effort length if a concrete class exposes a loaded dataset."""
        if hasattr(self, "dataset") and getattr(self, "dataset") is not None:
            return len(getattr(self, "dataset"))
        # Try to load and check again
        try:
            self.load()
        except Exception:
            pass
        if hasattr(self, "dataset") and getattr(self, "dataset") is not None:
            return len(getattr(self, "dataset"))
        raise NotImplementedError("Concrete benchmark must implement __len__ or expose dataset")
