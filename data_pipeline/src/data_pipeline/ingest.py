"""수집(변형1의 앞부분). 소스에서 데이터를 가져와 raw_articles에 pipeline_status='pending'으로
삽입한다. 비구조화 소스는 여기서 LLM으로 필드를 정제한다(구조화 소스는 그대로 매핑, 2절).
키워드 추출은 여기서 하지 않는다 — extract.py가 'pending' 상태 행을 대상으로 별도 처리한다
(요청 실패/중단 시에도 이미 수집된 행은 안전하게 남도록 단계를 분리했다).
"""
from __future__ import annotations

from datetime import date
from typing import Any

from . import db
from .llm_client import call_llm, parse_json_response
from .sources.base import DataSource

UNSTRUCTURED_EXTRACT_PROMPT = """다음 원문에서 title, description, source, published_at(ISO8601)을
추출해 JSON으로만 답하라: {{"title": "...", "description": "...", "source": "...", "published_at": "..."}}

# 원문
{raw_content}
"""


def _extract_fields_via_llm(raw_item: dict) -> dict:
    raw_content = raw_item.get("raw_content", "")
    raw = call_llm("당신은 정보 추출기입니다.", UNSTRUCTURED_EXTRACT_PROMPT.format(raw_content=raw_content))
    fields = parse_json_response(raw)
    fields["url"] = raw_item.get("url", "")  # url은 소스가 이미 알고 있는 값을 신뢰한다(LLM이 지어내지 않게)
    return fields


def run_ingest(source: DataSource, *, digest_date: str | None = None, **fetch_kwargs: Any) -> dict:
    digest_date = digest_date or date.today().isoformat()
    items = source.fetch(**fetch_kwargs)

    inserted, skipped, failed = 0, 0, 0
    for item in items:
        try:
            fields = item if source.is_structured else _extract_fields_via_llm(item)
            new_id = db.insert_raw_article(digest_date, fields, status="pending")
            if new_id is not None:
                inserted += 1
            else:
                skipped += 1  # 중복 url
        except Exception as exc:  # noqa: BLE001 - 개별 항목 실패가 전체 배치를 막지 않게 한다
            failed += 1
            print(f"[data_pipeline] ingest 실패 (source={source.name}): {exc}")

    return {"source": source.name, "fetched": len(items), "inserted": inserted, "skipped": skipped, "failed": failed}
