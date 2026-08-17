"""indexing stage: 기사 본문을 chunk/embedding과 GLiNER2 4-Layer로 저장한다.

`index_articles()`는 본문 수집부터 Event 저장까지 한 번에 처리하고, `index_events()`는
이미 저장된 `article_chunks`만 재료로 Event만 다시 인덱싱한다. SQL은 모두 `db.py`에 있다.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from config import config

from . import db
from .content import (
    ArticleContentDependencyError,
    build_article_text,
    fetch_article_body,
    load_embedding_tokenizer,
    split_text,
)
from .embed import embed_texts
from .extract import (
    GLiNER2EventExtractor,
    GLiNER2TopicExtractor,
    aggregate_event_candidates,
    load_event_extractor,
    load_gliner2_model,
)
from .taxonomy import (
    EVENT_EXTRACTION_VERSION,
    EVENT_TAXONOMY_VERSION,
    GLINER2_MODEL_NAME,
)


# ---------------------------------------------------------------------------
# 본문 + chunk/embedding + 4-Layer metadata
# ---------------------------------------------------------------------------

def _resolve_article_text(article: dict) -> tuple[str, str]:
    """URL 본문을 우선하고 어떤 수집 실패든 metadata fallback으로 격리한다."""
    try:
        body = fetch_article_body(article["url"])
    except ArticleContentDependencyError:
        raise
    except Exception:  # 뉴스 사이트별 차단/HTML 차이는 해당 기사 fallback으로 끝낸다.
        body = None

    text_source = "body" if body else "title+description"
    text = build_article_text(article["title"], article["description"], body)
    return text, text_source


def index_articles(
    article_ids: Iterable[int],
    *,
    device: str = "cuda",
) -> list[dict]:
    """명시한 기사의 본문·4-Layer metadata·embedding을 함께 인덱싱한다."""
    requested_ids = db.normalize_article_ids(article_ids)
    if not requested_ids:
        return []

    articles = db.load_articles(requested_ids)

    tokenizer = load_embedding_tokenizer()
    model = load_gliner2_model(device)
    topic_extractor = GLiNER2TopicExtractor(model)
    prepared: list[tuple[dict, list[dict], str, dict]] = []
    all_texts: list[str] = []
    for article in articles:
        text, text_source = _resolve_article_text(article)
        chunks = split_text(text, tokenizer)
        if not chunks:
            raise ValueError(f"article_id={article['id']}의 indexing 텍스트가 비어 있습니다.")
        topics = topic_extractor.extract(text)
        prepared.append((article, chunks, text_source, topics))
        all_texts.extend(chunk["chunk_text"] for chunk in chunks)

    all_vectors = embed_texts(all_texts)
    results: list[dict] = []
    offset = 0
    for article, chunks, text_source, topics in prepared:
        end = offset + len(chunks)
        db.store_article_index(article["id"], chunks, all_vectors[offset:end], topics)
        results.append(
            {
                "article_id": article["id"],
                "chunk_count": len(chunks),
                "text_source": text_source,
                "category": topics["category"],
                "domains": topics["domains"],
                "entities": topics["entities"],
                "embedding_model": config.HF_EMBEDDING_MODEL,
                "embedding_dimension": config.HF_EMBEDDING_DIMENSION,
            }
        )
        offset = end

    event_results = index_events(
        [result["article_id"] for result in results],
        device=device,
        extractor=GLiNER2EventExtractor(model),
    )
    events_by_article = {result["article_id"]: result for result in event_results}
    for result in results:
        event_result = events_by_article[result["article_id"]]
        result["event_status"] = event_result["status"]
        result["event_count"] = event_result["event_count"]
        if event_result.get("error"):
            result["event_error"] = event_result["error"]
    return results


def index_all_articles(limit: int = 20, *, device: str = "cuda") -> list[dict]:
    """embedding이 없는 기사 중 제한된 수만 본문 수집과 인덱싱을 수행한다."""
    if limit <= 0:
        raise ValueError("limit은 1 이상이어야 합니다.")
    return index_articles(db.load_unindexed_article_ids(limit), device=device)


# ---------------------------------------------------------------------------
# 구조화 Event
# ---------------------------------------------------------------------------

def _source_fingerprint(chunks: list[dict]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk["id"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(chunk["chunk_index"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(chunk["chunk_text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_current(status: dict | None, source_fingerprint: str) -> bool:
    return bool(
        status
        and status["status"] == "completed"
        and status["extractor_model"] == GLINER2_MODEL_NAME
        and status["taxonomy_version"] == EVENT_TAXONOMY_VERSION
        and status["extraction_version"] == EVENT_EXTRACTION_VERSION
        and status["source_fingerprint"] == source_fingerprint
    )


def index_events(
    article_ids: Iterable[int],
    *,
    force: bool = False,
    device: str = "cuda",
    extractor=None,
) -> list[dict]:
    """명시한 기사들의 기존 chunk를 구조화 Event로 인덱싱한다."""
    requested_ids = db.normalize_article_ids(article_ids)
    if not requested_ids:
        return []

    articles = db.load_articles_with_chunks(requested_ids)
    missing_chunk_ids = [
        article["article_id"] for article in articles if not article["chunks"]
    ]
    if missing_chunk_ids:
        raise ValueError(
            "article_chunks가 없는 article_id: "
            f"{missing_chunk_ids}. 먼저 rag.indexer.index_all_articles()를 실행해 주세요."
        )

    statuses = db.load_event_statuses(requested_ids)
    prepared = [
        (article, _source_fingerprint(article["chunks"])) for article in articles
    ]
    pending = [
        item
        for item in prepared
        if force or not _is_current(statuses.get(item[0]["article_id"]), item[1])
    ]

    active_extractor = None
    if pending:
        active_extractor = extractor or load_event_extractor(device)
    pending_ids = {article["article_id"] for article, _fingerprint in pending}
    results: list[dict] = []

    for article, source_fingerprint in prepared:
        article_id = article["article_id"]
        if article_id not in pending_ids:
            results.append(
                {
                    "article_id": article_id,
                    "title": article["title"],
                    "status": "skipped",
                    "event_count": statuses[article_id]["event_count"],
                }
            )
            continue

        try:
            candidates: list[dict] = []
            for chunk in article["chunks"]:
                for event in active_extractor.extract(chunk["chunk_text"]):
                    candidates.append({**event, "source_chunk_id": chunk["id"]})
            events = aggregate_event_candidates(candidates)
            db.replace_article_events(article_id, source_fingerprint, events)
        except Exception as exc:  # 기사 하나의 실패가 다음 기사를 막지 않게 한다.
            db.mark_event_failed(article_id, source_fingerprint, str(exc))
            results.append(
                {
                    "article_id": article_id,
                    "title": article["title"],
                    "status": "failed",
                    "event_count": 0,
                    "error": str(exc),
                }
            )
            continue

        results.append(
            {
                "article_id": article_id,
                "title": article["title"],
                "status": "completed",
                "event_count": len(events),
            }
        )
    return results
