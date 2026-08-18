"""변형2: 'extracted' 상태 raw_articles에 LLM으로 팩트 해석/분류 메타데이터를 생성해
enrichment에 원시값(raw_*)으로 저장하고 'enriched'로 전환한다 (design doc 3절).

category/domain/entity/event는 아직 정규화 전 원시값이다 — normalize.py(변형3)가
synonym_table 기준으로 정리한다.
"""
from __future__ import annotations

from . import db
from .llm_client import call_llm, parse_json_response

PROMPT_VERSION = "enrich_v1"

SYSTEM_PROMPT = """당신은 뉴스 기사를 분석해 학습 데이터용 메타데이터를 만드는 애널리스트다.
아래 JSON 형식으로만 답하라. 다른 설명은 붙이지 않는다.

```json
{
  "insights": [{"text": "인사이트 문장", "source_url": "기사 URL"}],
  "implications": ["비즈니스 시사점 문장"],
  "category": "대분류 (예: 경제/기술/산업/금융 중 하나, 자유 기술 가능)",
  "domain": ["세부 도메인 (예: 반도체, AI, 2차전지)"],
  "entity": ["관련 기업/기관"],
  "event": ["이벤트 유형 (예: 투자, 인수합병, 실적, 규제, 사고)"]
}
```
insights는 3~5개, category는 정확히 1개 문자열, domain/entity/event는 해당사항이 있는 만큼
배열로 채우고 없으면 빈 배열로 둔다."""


def _build_user_prompt(article: dict) -> str:
    return (
        f"# 제목\n{article['title']}\n\n"
        f"# 설명\n{article.get('description') or ''}\n\n"
        f"# URL\n{article['url']}\n\n"
        "위 기사를 분석해 메타데이터를 JSON으로 만들어줘."
    )


def enrich_article(article: dict) -> dict:
    raw = call_llm(SYSTEM_PROMPT, _build_user_prompt(article), max_tokens=1200)
    return parse_json_response(raw)


def run_enrich(limit: int) -> dict:
    from .config import config

    candidates = db.fetch_articles_by_status("extracted", limit)
    done, failed = 0, 0

    for article in candidates:
        try:
            result = enrich_article(article)
            db.insert_enrichment_raw(
                raw_article_id=article["id"],
                insights=result.get("insights", []),
                implications=result.get("implications", []),
                raw_category=result.get("category"),
                raw_domain=result.get("domain", []),
                raw_entity=result.get("entity", []),
                raw_event=result.get("event", []),
                model_used={
                    "groq": config.GROQ_MODEL,
                    "hf": config.HF_MODEL_ID,
                    "anthropic": config.ANTHROPIC_MODEL,
                }.get(config.LLM_PROVIDER, config.LLM_PROVIDER),
                prompt_version=PROMPT_VERSION,
            )
            done += 1
        except Exception as exc:  # noqa: BLE001
            db.mark_failed(article["id"], "enrich", str(exc))
            failed += 1

    return {"stage": "enrich", "candidates": len(candidates), "done": done, "failed": failed}
