from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SPIDER2_REPO_URL = "https://github.com/xlang-ai/Spider2.git"
SPIDER2_DBT_START_DB_ZIP = "DBT_start_db.zip"
SPIDER2_DBT_GOLD_ZIP = "dbt_gold.zip"


def is_spider2_dbt_setup_complete(spider2_dbt_dir: Path) -> bool:
    """
    Check if Spider2-DBT setup has been completed by looking for .duckdb files.

    Mirrors the setup checks from the standalone conversion script: we expect both
    example DBs and gold DBs to exist after running Spider2's `setup.py`.
    """
    spider2_dbt_dir = Path(spider2_dbt_dir)
    examples_dir = spider2_dbt_dir / "examples"
    gold_dir = spider2_dbt_dir / "evaluation_suite" / "gold"

    has_example_dbs = (
        any(examples_dir.rglob("*.duckdb")) if examples_dir.exists() else False
    )
    has_gold_dbs = any(gold_dir.rglob("*.duckdb")) if gold_dir.exists() else False

    return has_example_dbs and has_gold_dbs


def setup_spider2_dbt_from_cache_zips(*, spider2_dbt_dir: Path, cache_dir: Path) -> None:
    """
    Use pre-downloaded zip files from `cache_dir` and run upstream Spider2 `setup.py`.

    Expected files in `cache_dir`:
    - DBT_start_db.zip
    - dbt_gold.zip
    """
    spider2_dbt_dir = Path(spider2_dbt_dir)
    if not spider2_dbt_dir.is_dir():
        raise FileNotFoundError(f"Spider2-DBT directory not found: {spider2_dbt_dir}")

    cache_dir = Path(cache_dir)
    start_zip = cache_dir / SPIDER2_DBT_START_DB_ZIP
    if not start_zip.is_file():
        raise FileNotFoundError(
            f"Required zip not found: {start_zip} (expected exact name {SPIDER2_DBT_START_DB_ZIP})"
        )

    gold_zip = cache_dir / SPIDER2_DBT_GOLD_ZIP
    if not gold_zip.is_file():
        raise FileNotFoundError(
            f"Required zip not found: {gold_zip} (expected exact name {SPIDER2_DBT_GOLD_ZIP})"
        )

    # Upstream `setup.py` expects these zip names inside `spider2-dbt/`.
    logger.info("Copying Spider2-DBT zips from cache_dir=%s into %s", cache_dir, spider2_dbt_dir)
    shutil.copy2(start_zip, spider2_dbt_dir / SPIDER2_DBT_START_DB_ZIP)
    shutil.copy2(gold_zip, spider2_dbt_dir / SPIDER2_DBT_GOLD_ZIP)

    # Run setup.py
    logger.info("Running Spider2-DBT setup.py to extract databases...")
    subprocess.run(
        ["python", "setup.py"],
        cwd=str(spider2_dbt_dir),
        check=True,
    )


def ensure_spider2_dbt_root(
    *,
    cache_dir: Path,
    spider2_repo_url: str = SPIDER2_REPO_URL,
    force_reclone: bool = False,
) -> Path:
    """
    Ensure a usable `spider2-dbt` dataset directory exists locally.

    Behavior:
    - Checks for an existing clone at `cache_dir/Spider2`.
    - If missing (or `force_reclone=True`), clones Spider2.
    - Ensures `.../spider2-dbt` is fully set up; if not, copies the required zip
      files from `cache_dir` and runs upstream `setup.py`.
    """
    cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    spider2_repo_root = cache_dir / "Spider2"

    if spider2_repo_root.exists() and force_reclone:
        shutil.rmtree(spider2_repo_root, ignore_errors=True)

    if not spider2_repo_root.exists():
        logger.info("Cloning Spider2 repository into %s", spider2_repo_root)
        spider2_repo_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", spider2_repo_url, str(spider2_repo_root)],
            check=True,
        )

    root = spider2_repo_root / "spider2-dbt"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected spider2-dbt directory not found under: {spider2_repo_root} "
            f"(looked for {root})."
        )

    if not is_spider2_dbt_setup_complete(root):
        setup_spider2_dbt_from_cache_zips(spider2_dbt_dir=root, cache_dir=cache_dir)

    return root


