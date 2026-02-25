from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import duckdb
from word2word import Word2word

_ALTERNATE_TIMESTAMP_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%d-%b-%Y %H:%M:%S",
]


@dataclass(frozen=True)
class DuckDBPerturbationResult:
    db_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PerturbContext:
    timestamp_format: str
    rng: random.Random
    translate_probability: float
    translators: Dict[str, Optional[Word2word]]
    translation_languages: List[str]


@dataclass
class ColumnPlan:
    old_name: str
    decl_type: str
    expr: str
    alias: str
    ops: List[str] = field(default_factory=list)
    translated_language: Optional[str] = None


ColumnPerturbation = Callable[[ColumnPlan, PerturbContext], ColumnPlan]


def _quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _quote_schema_table(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _list_base_tables(con: duckdb.DuckDBPyConnection) -> List[Tuple[str, str]]:
    return [
        (schema, table)
        for schema, table in con.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            """
        ).fetchall()
    ]


def _list_columns(
    con: duckdb.DuckDBPyConnection,
    *,
    schema: str,
    table: str,
) -> List[Tuple[str, str]]:
    return [
        (name, dtype)
        for name, dtype in con.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema, table],
        ).fetchall()
    ]


def _is_timestamp_type(decl_type: str) -> bool:
    return "TIMESTAMP" in (decl_type or "").upper()


def _shuffle_header_tokens(name: str, rng: random.Random) -> str:
    if "_" not in name:
        return name
    tokens = [t for t in name.split("_") if t]
    if len(tokens) <= 1:
        return name

    shuffled = tokens[:]
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != tokens:
            return "_".join(shuffled)

    # Fallback for repeated failed shuffles (e.g., unlucky draws).
    rotated = tokens[1:] + tokens[:1]
    return "_".join(rotated)


def _perturb_timestamp_format(plan: ColumnPlan, ctx: PerturbContext) -> ColumnPlan:
    if _is_timestamp_type(plan.decl_type):
        plan.expr = f"strftime('{ctx.timestamp_format}', {plan.expr})"
        plan.ops.append("timestamp_format")
    return plan


def _perturb_header_shuffle(plan: ColumnPlan, ctx: PerturbContext) -> ColumnPlan:
    new_alias = _shuffle_header_tokens(plan.alias, ctx.rng)
    if new_alias != plan.alias:
        plan.alias = new_alias
        plan.ops.append("header_shuffle")
    return plan


def _translate_token(token: str, translator: Optional[Word2word]) -> str:
    if not token or translator is None:
        return token
    try:
        candidates = translator(token)
    except Exception:
        return token
    if not candidates:
        return token
    return str(candidates[0]) if candidates[0] else token


def _translate_header(name: str, translator: Optional[Word2word]) -> str:
    if not name:
        return name
    if "_" not in name:
        return _translate_token(name, translator)

    translated_tokens = [
        _translate_token(token, translator) if token else token
        for token in name.split("_")
    ]
    return "_".join(translated_tokens)


def _perturb_header_translate(plan: ColumnPlan, ctx: PerturbContext) -> ColumnPlan:
    if ctx.rng.random() >= ctx.translate_probability:
        return plan

    language = ctx.rng.choice(ctx.translation_languages)
    translator = ctx.translators.get(language)
    translated_alias = _translate_header(plan.alias, translator)
    if translated_alias != plan.alias:
        plan.alias = translated_alias
        plan.translated_language = language
        plan.ops.append("header_translate")
    return plan


def _apply_column_perturbations(
    cols: List[Tuple[str, str]],
    *,
    ctx: PerturbContext,
    perturbations: List[ColumnPerturbation],
) -> List[ColumnPlan]:
    plans = [
        ColumnPlan(
            old_name=name,
            decl_type=decl_type,
            expr=_quote_ident(name),
            alias=name,
        )
        for name, decl_type in cols
    ]

    for i, plan in enumerate(plans):
        for fn in perturbations:
            plan = fn(plan, ctx)
        plans[i] = plan
    return plans


def _dedupe_aliases(plans: List[ColumnPlan]) -> None:
    used: Dict[str, int] = {}
    for plan in plans:
        base = plan.alias
        idx = used.get(base, 0)
        plan.alias = base if idx == 0 else f"{base}_{idx}"
        used[base] = idx + 1


def _build_select_exprs(plans: List[ColumnPlan]) -> List[str]:
    return [f"{p.expr} AS {_quote_ident(p.alias)}" for p in plans]


def _rewrite_table_in_place(
    con: duckdb.DuckDBPyConnection,
    *,
    schema: str,
    table: str,
    select_exprs: List[str],
    rng: random.Random,
) -> None:
    source_fq = _quote_schema_table(schema, table)
    tmp_name = f"__ca_pert_{table}_{rng.randrange(1_000_000)}"
    tmp_fq = _quote_schema_table(schema, tmp_name)

    con.execute(f"DROP TABLE IF EXISTS {tmp_fq}")
    con.execute(
        f"CREATE TABLE {tmp_fq} AS "
        f"SELECT {', '.join(select_exprs)} FROM {source_fq}"
    )
    con.execute(f"DROP TABLE {source_fq}")
    con.execute(f"ALTER TABLE {tmp_fq} RENAME TO {_quote_ident(table)}")


def perturb_duckdb(
    db_path: str | Path,
    *,
    spec: Dict[str, Any],
    seed: int,
) -> DuckDBPerturbationResult:
    """Apply timestamp-format, header-shuffle, and random-language header translation."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    rng = random.Random(seed)
    name = str(spec.get("name") or "timestamp_header_perturb")
    configured_timestamp_format = spec.get("timestamp_format") or (
        (spec.get("values") or {}).get("timestamp_format")
    )
    timestamp_format = (
        str(configured_timestamp_format)
        if configured_timestamp_format
        else rng.choice(_ALTERNATE_TIMESTAMP_FORMATS)
    )
    configured_translate_probability = spec.get("translate_probability") or (
        (spec.get("values") or {}).get("translate_probability")
    )
    translate_probability = (
        float(configured_translate_probability)
        if configured_translate_probability is not None
        else 0.5
    )
    translate_probability = max(0.0, min(1.0, translate_probability))

    translation_languages = ["fr", "zh", "ja", "es"]
    word2word_lang_codes = {
        "fr": "fr",
        "zh": "zh_cn",
        "ja": "ja",
        "es": "es",
    }
    translators: Dict[str, Optional[Word2word]] = {}
    for language in translation_languages:
        code = word2word_lang_codes[language]
        try:
            translators[language] = Word2word("en", code)
        except Exception:
            translators[language] = None

    ctx = PerturbContext(
        timestamp_format=timestamp_format,
        rng=rng,
        translate_probability=translate_probability,
        translators=translators,
        translation_languages=translation_languages,
    )
    perturbations: List[ColumnPerturbation] = [
        _perturb_timestamp_format,
        _perturb_header_shuffle,
        _perturb_header_translate,
    ]

    manifest: Dict[str, Any] = {
        "name": name,
        "timestamp_format": timestamp_format,
        "translate_probability": translate_probability,
        "translation_languages": translation_languages,
        "tables": {},
    }

    print(f"[dbpert] database: {db_path}", flush=True)
    con = duckdb.connect(database=str(db_path))
    try:
        for schema, table in _list_base_tables(con):
            cols = _list_columns(con, schema=schema, table=table)
            if not cols:
                continue

            plans = _apply_column_perturbations(
                cols,
                ctx=ctx,
                perturbations=perturbations,
            )
            _dedupe_aliases(plans)

            select_exprs = _build_select_exprs(plans)
            changed = any(p.ops or p.old_name != p.alias for p in plans)
            if changed:
                _rewrite_table_in_place(
                    con,
                    schema=schema,
                    table=table,
                    select_exprs=select_exprs,
                    rng=rng,
                )

            manifest["tables"][f"{schema}.{table}"] = {
                "timestamp_columns": [p.old_name for p in plans if "timestamp_format" in p.ops],
                "translated_columns": [p.old_name for p in plans if "header_translate" in p.ops],
                "translated_column_languages": {
                    p.old_name: p.translated_language
                    for p in plans
                    if "header_translate" in p.ops and p.translated_language is not None
                },
                "column_mapping": {
                    p.old_name: p.alias for p in plans if p.old_name != p.alias
                },
            }
    finally:
        con.close()

    manifest_path = db_path.with_name(f"{db_path.name}.{name}.perturbation_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return DuckDBPerturbationResult(db_path=db_path, manifest_path=manifest_path)
