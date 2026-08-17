"""기존 article_chunks에서 GLiNER2 구조화 Event를 추출해 저장한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable

from db.db import get_conn, init_db

from .event_extractor import aggregate_event_candidates, load_event_extractor
from .event_taxonomy import EXTRACTION_VERSION, MODEL_NAME, TAXONOMY_VERSION


def _load_articles_with_chunks(article_ids: list[int]) -> list[dict]:
    with get_conn() as conn:
        article_rows = conn.execute(
            """SELECT id, title
               FROM raw_articles
               WHERE id = ANY(%s)
               ORDER BY id""",
            (article_ids,),
        ).fetchall()
        chunk_rows = conn.execute(
            """SELECT id, article_id, chunk_index, chunk_text
               FROM article_chunks
               WHERE article_id = ANY(%s)
               ORDER BY article_id, chunk_index""",
            (article_ids,),
        ).fetchall()

    chunks_by_article: dict[int, list[dict]] = {}
    for chunk in chunk_rows:
        chunks_by_article.setdefault(chunk["article_id"], []).append(chunk)
    return [
        {
            "article_id": article["id"],
            "title": article["title"],
            "chunks": chunks_by_article.get(article["id"], []),
        }
        for article in article_rows
    ]


def _load_statuses(article_ids: list[int]) -> dict[int, dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT article_id, extractor_model, taxonomy_version,
                      extraction_version, source_fingerprint, status, event_count
               FROM article_event_index_status
               WHERE article_id = ANY(%s)""",
            (article_ids,),
        ).fetchall()
    return {row["article_id"]: row for row in rows}


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
        and status["extractor_model"] == MODEL_NAME
        and status["taxonomy_version"] == TAXONOMY_VERSION
        and status["extraction_version"] == EXTRACTION_VERSION
        and status["source_fingerprint"] == source_fingerprint
    )


def _replace_article_events(
    article_id: int,
    source_fingerprint: str,
    events: list[dict],
) -> None:
    """한 기사 결과 전체와 completed 상태를 같은 transaction에서 교체한다."""
    with get_conn() as conn:
        conn.execute("DELETE FROM article_events WHERE article_id = %s", (article_id,))

        for event in events:
            event_id = conn.execute(
                """INSERT INTO article_events
                       (article_id, source_chunk_id, event_type, confidence,
                        extractor_model, taxonomy_version, extraction_version,
                        event_fingerprint, evidence_start, evidence_end)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    article_id,
                    event["source_chunk_id"],
                    event["event_type"],
                    event["confidence"],
                    MODEL_NAME,
                    TAXONOMY_VERSION,
                    EXTRACTION_VERSION,
                    event["event_fingerprint"],
                    event["evidence_start"],
                    event["evidence_end"],
                ),
            ).fetchone()["id"]

            for argument_index, argument in enumerate(event["arguments"]):
                conn.execute(
                    """INSERT INTO article_event_arguments
                           (event_id, argument_index, role, text, normalized_text,
                            confidence, span_start, span_end)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        event_id,
                        argument_index,
                        argument["role"],
                        argument["text"],
                        argument["normalized_text"],
                        argument["confidence"],
                        argument["span_start"],
                        argument["span_end"],
                    ),
                )

        conn.execute(
            """INSERT INTO article_event_index_status
                   (article_id, extractor_model, taxonomy_version, extraction_version,
                    source_fingerprint, status, event_count, error, processed_at)
               VALUES (%s, %s, %s, %s, %s, 'completed', %s, NULL, now())
               ON CONFLICT (article_id) DO UPDATE SET
                   extractor_model = EXCLUDED.extractor_model,
                   taxonomy_version = EXCLUDED.taxonomy_version,
                   extraction_version = EXCLUDED.extraction_version,
                   source_fingerprint = EXCLUDED.source_fingerprint,
                   status = EXCLUDED.status,
                   event_count = EXCLUDED.event_count,
                   error = NULL,
                   processed_at = now()""",
            (
                article_id,
                MODEL_NAME,
                TAXONOMY_VERSION,
                EXTRACTION_VERSION,
                source_fingerprint,
                len(events),
            ),
        )


def _mark_failed(article_id: int, source_fingerprint: str, error: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO article_event_index_status
                   (article_id, extractor_model, taxonomy_version, extraction_version,
                    source_fingerprint, status, event_count, error, processed_at)
               VALUES (%s, %s, %s, %s, %s, 'failed', 0, %s, now())
               ON CONFLICT (article_id) DO UPDATE SET
                   extractor_model = EXCLUDED.extractor_model,
                   taxonomy_version = EXCLUDED.taxonomy_version,
                   extraction_version = EXCLUDED.extraction_version,
                   source_fingerprint = EXCLUDED.source_fingerprint,
                   status = EXCLUDED.status,
                   event_count = 0,
                   error = EXCLUDED.error,
                   processed_at = now()""",
            (
                article_id,
                MODEL_NAME,
                TAXONOMY_VERSION,
                EXTRACTION_VERSION,
                source_fingerprint,
                error,
            ),
        )


def index_event_articles(
    article_ids: Iterable[int],
    *,
    force: bool = False,
    device: str = "cuda",
) -> list[dict]:
    """명시한 기사들의 기존 chunk를 구조화 Event로 인덱싱한다."""
    requested_ids = list(dict.fromkeys(int(article_id) for article_id in article_ids))
    if not requested_ids:
        return []

    articles = _load_articles_with_chunks(requested_ids)
    found_ids = {article["article_id"] for article in articles}
    missing_ids = [article_id for article_id in requested_ids if article_id not in found_ids]
    if missing_ids:
        raise ValueError(f"존재하지 않는 article_id: {missing_ids}")

    missing_chunk_ids = [
        article["article_id"] for article in articles if not article["chunks"]
    ]
    if missing_chunk_ids:
        raise ValueError(
            "article_chunks가 없는 article_id: "
            f"{missing_chunk_ids}. 먼저 rag.indexer.index_all_articles()를 실행해 주세요."
        )

    statuses = _load_statuses(requested_ids)
    prepared = [
        (article, _source_fingerprint(article["chunks"])) for article in articles
    ]
    pending = [
        item
        for item in prepared
        if force or not _is_current(statuses.get(item[0]["article_id"]), item[1])
    ]

    extractor = load_event_extractor(device) if pending else None
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
                for event in extractor.extract(chunk["chunk_text"]):
                    candidates.append({**event, "source_chunk_id": chunk["id"]})
            events = aggregate_event_candidates(candidates)
            _replace_article_events(article_id, source_fingerprint, events)
        except Exception as exc:  # 기사 하나의 실패가 다음 기사를 막지 않게 한다.
            _mark_failed(article_id, source_fingerprint, str(exc))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="기존 article chunk의 구조화 Event indexing")
    parser.add_argument("--article-ids", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    init_db()
    results = index_event_articles(
        args.article_ids,
        force=args.force,
        device=args.device,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
