"""4단계(ingest -> extract -> enrich -> normalize)를 순서대로 실행하는 오케스트레이션.

각 단계는 독립적으로도 실행 가능하다(cli.py --stage). run_all은 한 프로세스 안에서
순서대로 이어 돌리는 편의 함수로, 배치 한 번에 처리할 건수(config.BATCH_LIMIT)만큼만
처리하고 결과를 반환한다 — 무료 API 환경에서 중간에 끊겨도 다음 실행이 이어받을 수 있도록
욕심내지 않고 작은 배치로 도는 것을 기본으로 한다.
"""
from __future__ import annotations

import time

from . import extract, ingest, normalize, enrich as enrich_stage
from .config import config
from .sources.base import DataSource
from .sources.gnews_source import GNewsSource


def run_ingest_multi(keywords: list[str], *, source: DataSource | None = None, **fetch_kwargs) -> dict:
    """여러 키워드로 나눠 ingest를 돌리고 결과를 합산한다. 키워드별로 별도
    요청을 보내므로(GNews/네이버 모두 다중 키워드 OR 검색을 지원하지 않음), 카테고리를
    넓게 커버하고 싶을 때(예: 경제/산업/금융/기술) 이걸 쓴다. source를 안 주면 기존과
    동일하게 GNews를 쓴다 — 네이버로 돌리려면 `source=NaverNewsSource()`.

    무료/개발자 플랜은 초당 요청 수 제한이 있다 — 키워드를 곧바로 이어서 요청하면
    429가 난다(실제로 겪음). 키워드 사이에 살짝 텀을 둔다."""
    source = source or GNewsSource()
    per_keyword = []
    totals = {"fetched": 0, "inserted": 0, "skipped": 0, "failed": 0}
    for i, kw in enumerate(keywords):
        if i > 0:
            time.sleep(1.2)
        result = ingest.run_ingest(source, keyword=kw, **fetch_kwargs)
        per_keyword.append(result)
        for key in totals:
            totals[key] += result[key]
    return {"source": source.name, "keywords": keywords, **totals, "per_keyword": per_keyword}


def run_all(*, limit: int | None = None, do_ingest: bool = True, keywords: list[str] | None = None) -> dict:
    limit = limit or config.BATCH_LIMIT
    results = {}

    if do_ingest:
        results["ingest"] = run_ingest_multi(keywords) if keywords else ingest.run_ingest(GNewsSource())

    results["extract"] = extract.run_extract(limit)
    results["enrich"] = enrich_stage.run_enrich(limit)
    results["normalize"] = normalize.run_normalize(limit)
    return results
