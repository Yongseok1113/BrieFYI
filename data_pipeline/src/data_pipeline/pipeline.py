"""4단계(ingest -> extract -> enrich -> normalize)를 순서대로 실행하는 오케스트레이션.

각 단계는 독립적으로도 실행 가능하다(cli.py --stage). run_all은 한 프로세스 안에서
순서대로 이어 돌리는 편의 함수로, 배치 한 번에 처리할 건수(config.BATCH_LIMIT)만큼만
처리하고 결과를 반환한다 — 무료 API 환경에서 중간에 끊겨도 다음 실행이 이어받을 수 있도록
욕심내지 않고 작은 배치로 도는 것을 기본으로 한다.
"""
from __future__ import annotations

from . import extract, ingest, normalize, enrich as enrich_stage
from .config import config
from .sources.gnews_source import GNewsSource


def run_all(*, limit: int | None = None, do_ingest: bool = True) -> dict:
    limit = limit or config.BATCH_LIMIT
    results = {}

    if do_ingest:
        results["ingest"] = ingest.run_ingest(GNewsSource())

    results["extract"] = extract.run_extract(limit)
    results["enrich"] = enrich_stage.run_enrich(limit)
    results["normalize"] = normalize.run_normalize(limit)
    return results
