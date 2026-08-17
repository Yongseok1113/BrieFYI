"""통합 단어 테이블(synonym_table) 자동 생성/갱신 (design doc 4절).

파이프라인의 실시간 흐름(ingest -> extract -> enrich -> normalize)과 분리된 배치 작업이다.
사람 검수는 기본적으로 막지 않는다 — 자동 생성 결과를 바로 synonym_table에 반영하고,
reviewed=false로 표시해두어 나중에 scripts/run_synonym_builder.py --review 같은 별도
작업으로 검수할 수 있게만 해둔다 (요청대로 검수가 파이프라인을 막지 않는다).
"""
from __future__ import annotations

from . import db
from .clustering import cluster_values, default_embed_fn
from .config import config

DIMENSIONS = ("category", "domain", "entity", "event")


def build_dimension(dimension: str, *, embed_fn=default_embed_fn) -> dict:
    values_with_counts = db.fetch_raw_value_counts(dimension)
    clusters = cluster_values(values_with_counts, config.SYNONYM_CLUSTER_THRESHOLD, embed_fn)

    for cluster in clusters:
        db.upsert_synonym_entry(dimension, cluster.canonical, cluster.aliases, reviewed=False)

    return {"dimension": dimension, "raw_values": len(values_with_counts), "clusters": len(clusters)}


def build_all(*, embed_fn=default_embed_fn) -> list[dict]:
    return [build_dimension(dim, embed_fn=embed_fn) for dim in DIMENSIONS]
