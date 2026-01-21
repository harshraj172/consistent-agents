from .base import BaseBenchmark
from .swebench.swebench_benchmark import SWEBenchBenchmark
from .harbor_swebench import HarborSWEBenchBenchmark
from .harbor_spider2_dbt import HarborSpider2DBTBenchmark
from .truthfulqa.truthfulqa_benchmark import TruthfulQABenchmark

try:
    from .swebench.swebench_benchmark import SWEBenchBenchmark
except (ImportError, ModuleNotFoundError):
    SWEBenchBenchmark = None  # 'resource' unavailable on windows; check for non-WSL environment

try:
    from .harbor_spider2_dbt import HarborSpider2DBTBenchmark
except (ImportError, ModuleNotFoundError):
    HarborSpider2DBTBenchmark = None

__all__ = [
    "BaseBenchmark",
    "TruthfulQABenchmark",
    "SWEBenchBenchmark",
    "HarborSWEBenchBenchmark",
    "HarborSpider2DBTBenchmark",
]
