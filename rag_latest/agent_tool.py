"""LLM tool-use용 뉴스 검색 도구.

rag_experiment/rag/agent_search.py + agent_tool_schema.py를 rag_latest.retriever.retrieve()와
rag_latest.taxonomy의 실제 CATEGORY_LABELS/DOMAIN_LABELS에 맞춰 재작성했다.
rag_experiment/rag/classify.py의 자체 taxonomy는 쓰지 않는다 — DB에 실제로 저장되는
값(GLiNER2 기반 taxonomy.py)과 다른 enum을 노출하면 필터가 항상 빈 결과를 낼 수 있다.
"""
from __future__ import annotations

from typing import Any

from . import reranker, retriever, taxonomy

SEARCH_NEWS_TOOL_SCHEMA = {
    "name": "search_news",
    "description": "저장된 뉴스 기사에서 질의와 관련된 chunk를 검색한다. category/domain으로 좁혀 검색할 수 있다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "검색할 자연어 질의"},
            "top_k": {"type": "integer", "description": "반환할 결과 개수", "default": 5},
            "category": {
                "type": "string",
                "enum": list(taxonomy.CATEGORY_LABELS),
                "description": "결과를 좁힐 대분류(선택)",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string", "enum": list(taxonomy.DOMAIN_LABELS)},
                "description": "결과를 좁힐 세부 도메인(선택, 복수 가능)",
            },
        },
        "required": ["query"],
    },
}


def search_news(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    """검색 -> rerank까지 수행하고 LLM에 돌려줄 수 있게 직렬화된 결과를 반환한다."""
    candidates = retriever.retrieve(
        query=query,
        top_k=max(top_k * 3, top_k),
        search_mode="hybrid",
        category=category,
        domains=domains,
    )
    reranked = reranker.rerank(query, candidates, top_k=top_k)
    reranker.unload_reranker()

    return [
        {
            "article_id": row["article_id"],
            "title": row["title"],
            "url": row["url"],
            "text": row["text"][:500],
            "category": row.get("category"),
            "domains": row.get("domains") or [],
            "rerank_score": round(row["rerank_score"], 4),
        }
        for row in reranked
    ]


def execute_search_news(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Anthropic tool-use 루프에서 SEARCH_NEWS_TOOL_SCHEMA 호출 결과를 그대로 실행하는 dispatch shim."""
    return search_news(
        query=arguments["query"],
        top_k=arguments.get("top_k", 5),
        category=arguments.get("category"),
        domains=arguments.get("domains"),
    )
