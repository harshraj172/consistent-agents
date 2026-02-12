from .base import BaseBenchmark
from .swebench.swebench_benchmark import SWEBenchBenchmark
from .harbor_swebench import HarborSWEBenchBenchmark
from .truthfulqa.truthfulqa_benchmark import TruthfulQABenchmark

try:
    from .swebench.swebench_benchmark import SWEBenchBenchmark
except (ImportError, ModuleNotFoundError):
    SWEBenchBenchmark = None  # 'resource' unavailable on windows; check for non-WSL environment

try:
    from .bfcl import BFCLBenchmark
except (ImportError, ModuleNotFoundError):
    BFCLBenchmark = None

try:
    from .harbor_bfcl import HarborBFCLBenchmark
except (ImportError, ModuleNotFoundError):
    HarborBFCLBenchmark = None

__all__ = ['BaseBenchmark', 'TruthfulQABenchmark', 'SWEBenchBenchmark', 'HarborSWEBenchBenchmark', 'BFCLBenchmark', 'HarborBFCLBenchmark']
