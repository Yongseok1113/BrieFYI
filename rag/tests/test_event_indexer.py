"""실제 GLiNER2 호출 없이 Event indexer와 저장 계약을 검증한다."""
import unittest
from unittest import mock

from db.db import get_conn
from rag.event_indexer import (
    _replace_article_events,
    _source_fingerprint,
    index_event_articles,
)
from rag.event_taxonomy import EXTRACTION_VERSION, MODEL_NAME, TAXONOMY_VERSION
from tests.dbhelpers import TEST_URL_PREFIX, DbTestCase, requires_db


def _article(article_id=3, *, chunk_id=30, text="NVIDIA가 OpenAI에 투자했다."):
    return {
        "article_id": article_id,
        "title": "투자 기사",
        "chunks": [
            {
                "id": chunk_id,
                "article_id": article_id,
                "chunk_index": 0,
                "chunk_text": text,
            }
        ],
    }


def _event(confidence=0.9, *, source_chunk_id=None):
    event = {
        "event_type": "invested_in",
        "confidence": confidence,
        "event_fingerprint": "event-fingerprint",
        "evidence_start": 0,
        "evidence_end": 18,
        "arguments": [
            {
                "role": "investor",
                "text": "NVIDIA가",
                "normalized_text": "NVIDIA",
                "confidence": confidence,
                "span_start": 0,
                "span_end": 7,
            },
            {
                "role": "investee",
                "text": "OpenAI에",
                "normalized_text": "OpenAI",
                "confidence": confidence,
                "span_start": 8,
                "span_end": 15,
            },
        ],
    }
    if source_chunk_id is not None:
        event["source_chunk_id"] = source_chunk_id
    return event


class EventIndexerTest(unittest.TestCase):
    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses")
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_누락_ID를_model_load_전에_거부한다(self, load_articles, load_statuses, load_model):
        load_articles.return_value = [_article(3)]

        with self.assertRaisesRegex(ValueError, r"\[9\]"):
            index_event_articles([3, 9])

        load_statuses.assert_not_called()
        load_model.assert_not_called()

    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses")
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_chunk가_없는_ID를_model_load_전에_거부한다(
        self, load_articles, load_statuses, load_model
    ):
        article = _article(3)
        article["chunks"] = []
        load_articles.return_value = [article]

        with self.assertRaisesRegex(ValueError, "article_chunks"):
            index_event_articles([3])

        load_statuses.assert_not_called()
        load_model.assert_not_called()

    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses")
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_같은_입력과_version의_completed_기사를_skip한다(
        self, load_articles, load_statuses, load_model
    ):
        article = _article(3)
        fingerprint = _source_fingerprint(article["chunks"])
        load_articles.return_value = [article]
        load_statuses.return_value = {
            3: {
                "article_id": 3,
                "extractor_model": MODEL_NAME,
                "taxonomy_version": TAXONOMY_VERSION,
                "extraction_version": EXTRACTION_VERSION,
                "source_fingerprint": fingerprint,
                "status": "completed",
                "event_count": 2,
            }
        }

        result = index_event_articles([3])

        self.assertEqual("skipped", result[0]["status"])
        self.assertEqual(2, result[0]["event_count"])
        load_model.assert_not_called()

    @mock.patch("rag.event_indexer._replace_article_events")
    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses", return_value={})
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_Event가_0개여도_completed로_저장한다(
        self, load_articles, _load_statuses, load_model, replace_events
    ):
        article = _article(3)
        load_articles.return_value = [article]
        extractor = load_model.return_value
        extractor.extract.return_value = []

        result = index_event_articles([3], device="cpu")

        self.assertEqual("completed", result[0]["status"])
        self.assertEqual(0, result[0]["event_count"])
        replace_events.assert_called_once_with(
            3,
            _source_fingerprint(article["chunks"]),
            [],
        )
        load_model.assert_called_once_with("cpu")

    @mock.patch("rag.event_indexer._replace_article_events")
    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses")
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_extraction_version이_다르면_재처리한다(
        self, load_articles, load_statuses, load_model, replace_events
    ):
        article = _article(3)
        load_articles.return_value = [article]
        load_statuses.return_value = {
            3: {
                "article_id": 3,
                "extractor_model": MODEL_NAME,
                "taxonomy_version": TAXONOMY_VERSION,
                "extraction_version": "older-version",
                "source_fingerprint": _source_fingerprint(article["chunks"]),
                "status": "completed",
                "event_count": 1,
            }
        }
        load_model.return_value.extract.return_value = []

        result = index_event_articles([3])

        self.assertEqual("completed", result[0]["status"])
        replace_events.assert_called_once()

    @mock.patch("rag.event_indexer._replace_article_events")
    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses", return_value={})
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_여러_parent_chunk의_같은_Event는_최고점_하나만_저장한다(
        self, load_articles, _load_statuses, load_model, replace_events
    ):
        article = _article(3, chunk_id=30)
        article["chunks"].append(
            {
                "id": 31,
                "article_id": 3,
                "chunk_index": 1,
                "chunk_text": "OpenAI 투자 후속 설명",
            }
        )
        load_articles.return_value = [article]
        load_model.return_value.extract.side_effect = [[_event(0.91)], [_event(0.72)]]

        index_event_articles([3])

        stored_events = replace_events.call_args.args[2]
        self.assertEqual(1, len(stored_events))
        self.assertEqual(30, stored_events[0]["source_chunk_id"])
        self.assertAlmostEqual(0.91, stored_events[0]["confidence"])

    @mock.patch("rag.event_indexer._mark_failed")
    @mock.patch("rag.event_indexer._replace_article_events")
    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses", return_value={})
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_기사별_실패를_기록하고_다음_기사를_계속한다(
        self,
        load_articles,
        _load_statuses,
        load_model,
        replace_events,
        mark_failed,
    ):
        first = _article(3, chunk_id=30)
        second = _article(7, chunk_id=70)
        load_articles.return_value = [first, second]
        extractor = load_model.return_value
        extractor.extract.side_effect = [RuntimeError("model error"), []]

        result = index_event_articles([3, 7])

        self.assertEqual(["failed", "completed"], [row["status"] for row in result])
        mark_failed.assert_called_once_with(
            3,
            _source_fingerprint(first["chunks"]),
            "model error",
        )
        replace_events.assert_called_once_with(
            7,
            _source_fingerprint(second["chunks"]),
            [],
        )

    @mock.patch("rag.event_indexer._replace_article_events")
    @mock.patch("rag.event_indexer.load_event_extractor")
    @mock.patch("rag.event_indexer._load_statuses")
    @mock.patch("rag.event_indexer._load_articles_with_chunks")
    def test_force는_completed_기사도_재처리한다(
        self, load_articles, load_statuses, load_model, replace_events
    ):
        article = _article(3)
        load_articles.return_value = [article]
        load_statuses.return_value = {
            3: {
                "article_id": 3,
                "extractor_model": MODEL_NAME,
                "taxonomy_version": TAXONOMY_VERSION,
                "extraction_version": EXTRACTION_VERSION,
                "source_fingerprint": _source_fingerprint(article["chunks"]),
                "status": "completed",
                "event_count": 1,
            }
        }
        load_model.return_value.extract.return_value = []

        result = index_event_articles([3], force=True)

        self.assertEqual("completed", result[0]["status"])
        replace_events.assert_called_once()


@requires_db
class EventIndexerDbTest(DbTestCase):
    def test_Event_Argument_Status_저장과_cascade를_검증한다(self):
        with get_conn() as conn:
            article_id = conn.execute(
                """INSERT INTO raw_articles (digest_date, title, description, url)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (
                    self.today,
                    "구조화 Event 테스트 기사",
                    "NVIDIA가 OpenAI에 투자했다.",
                    TEST_URL_PREFIX + "event-indexer",
                ),
            ).fetchone()["id"]
            chunk_id = conn.execute(
                """INSERT INTO article_chunks (article_id, chunk_index, chunk_text)
                   VALUES (%s, 0, %s) RETURNING id""",
                (article_id, "NVIDIA가 OpenAI에 투자했다."),
            ).fetchone()["id"]

        _replace_article_events(
            article_id,
            "source-fingerprint",
            [_event(source_chunk_id=chunk_id)],
        )

        with get_conn() as conn:
            event = conn.execute(
                """SELECT event_type, event_count, status
                   FROM article_events e
                   JOIN article_event_index_status s ON s.article_id = e.article_id
                   WHERE e.article_id = %s""",
                (article_id,),
            ).fetchone()
            arguments = conn.execute(
                """SELECT role, text, normalized_text
                   FROM article_event_arguments a
                   JOIN article_events e ON e.id = a.event_id
                   WHERE e.article_id = %s
                   ORDER BY argument_index""",
                (article_id,),
            ).fetchall()

        self.assertEqual("invested_in", event["event_type"])
        self.assertEqual("completed", event["status"])
        self.assertEqual(1, event["event_count"])
        self.assertEqual(["investor", "investee"], [row["role"] for row in arguments])

        with get_conn() as conn:
            conn.execute("DELETE FROM raw_articles WHERE id = %s", (article_id,))
            event_count = conn.execute(
                "SELECT count(*) AS count FROM article_events WHERE article_id = %s",
                (article_id,),
            ).fetchone()["count"]
            status_count = conn.execute(
                "SELECT count(*) AS count FROM article_event_index_status WHERE article_id = %s",
                (article_id,),
            ).fetchone()["count"]
        self.assertEqual(0, event_count)
        self.assertEqual(0, status_count)


if __name__ == "__main__":
    unittest.main()
