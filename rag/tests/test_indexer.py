"""실제 DB, HF, GLiNER2 없이 indexing stage의 계약을 검증한다."""
import unittest
from unittest import mock

from rag.content import ArticleContentDependencyError
from rag.indexer import _source_fingerprint, index_all_articles, index_articles, index_events
from rag.taxonomy import (
    EVENT_EXTRACTION_VERSION,
    EVENT_TAXONOMY_VERSION,
    GLINER2_MODEL_NAME,
)
from rag.tests.test_content import OffsetTokenizer


def _article_with_chunks(article_id=3, *, chunk_id=30, text="NVIDIA가 OpenAI에 투자했다."):
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


def _completed_status(article_id, fingerprint, *, event_count=1, **overrides):
    return {
        article_id: {
            "article_id": article_id,
            "extractor_model": GLINER2_MODEL_NAME,
            "taxonomy_version": EVENT_TAXONOMY_VERSION,
            "extraction_version": EVENT_EXTRACTION_VERSION,
            "source_fingerprint": fingerprint,
            "status": "completed",
            "event_count": event_count,
        }
        | overrides
    }


class IndexArticlesTest(unittest.TestCase):
    ARTICLE = {
        "id": 3,
        "title": "기사 제목",
        "description": "짧은 설명",
        "url": "https://example.com/article",
    }

    TOPICS = {
        "category": "기술",
        "domains": ["AI"],
        "entities": ["OpenAI"],
    }

    def test_본문_수집에_성공하면_본문을_청킹한다(self):
        with (
            mock.patch("rag.indexer.db.load_articles", return_value=[self.ARTICLE]),
            mock.patch("rag.indexer.load_embedding_tokenizer", return_value=OffsetTokenizer()),
            mock.patch("rag.indexer.load_gliner2_model"),
            mock.patch("rag.indexer.GLiNER2TopicExtractor") as topic_extractor,
            mock.patch("rag.indexer.GLiNER2EventExtractor"),
            mock.patch("rag.indexer.fetch_article_body", return_value="실제 기사 본문"),
            mock.patch("rag.indexer.embed_texts", return_value=[[1.0, 0.0, 0.0]]) as embed,
            mock.patch("rag.indexer.db.store_article_index") as store,
            mock.patch(
                "rag.indexer.index_events",
                return_value=[{"article_id": 3, "status": "completed", "event_count": 1}],
            ),
        ):
            topic_extractor.return_value.extract.return_value = self.TOPICS
            result = index_articles([3])

        embed.assert_called_once_with(["기사 제목\n\n실제 기사 본문"])
        self.assertEqual(3, store.call_args.args[0])
        self.assertEqual(self.TOPICS, store.call_args.args[3])
        self.assertEqual("body", result[0]["text_source"])
        self.assertEqual("completed", result[0]["event_status"])

    def test_본문_수집에_실패하면_title_description을_사용한다(self):
        with (
            mock.patch("rag.indexer.db.load_articles", return_value=[self.ARTICLE]),
            mock.patch("rag.indexer.load_embedding_tokenizer", return_value=OffsetTokenizer()),
            mock.patch("rag.indexer.load_gliner2_model"),
            mock.patch("rag.indexer.GLiNER2TopicExtractor") as topic_extractor,
            mock.patch("rag.indexer.GLiNER2EventExtractor"),
            mock.patch("rag.indexer.fetch_article_body", side_effect=RuntimeError("blocked")),
            mock.patch("rag.indexer.embed_texts", return_value=[[1.0, 0.0, 0.0]]) as embed,
            mock.patch("rag.indexer.db.store_article_index"),
            mock.patch(
                "rag.indexer.index_events",
                return_value=[{"article_id": 3, "status": "completed", "event_count": 0}],
            ),
        ):
            topic_extractor.return_value.extract.return_value = self.TOPICS
            result = index_articles([3])

        embed.assert_called_once_with(["기사 제목\n\n짧은 설명"])
        self.assertEqual("title+description", result[0]["text_source"])

    def test_본문_추출_의존성_누락은_fallback으로_숨기지_않는다(self):
        with (
            mock.patch("rag.indexer.db.load_articles", return_value=[self.ARTICLE]),
            mock.patch("rag.indexer.load_embedding_tokenizer", return_value=OffsetTokenizer()),
            mock.patch("rag.indexer.load_gliner2_model"),
            mock.patch("rag.indexer.GLiNER2TopicExtractor"),
            mock.patch(
                "rag.indexer.fetch_article_body",
                side_effect=ArticleContentDependencyError("beautifulsoup4 missing"),
            ),
        ):
            with self.assertRaises(ArticleContentDependencyError):
                index_articles([3])


class IndexAllArticlesTest(unittest.TestCase):
    @mock.patch("rag.indexer.index_articles")
    @mock.patch("rag.indexer.db.load_unindexed_article_ids", return_value=[3, 7])
    def test_embedding이_없는_기사만_넘긴다(self, load_ids, index_articles_mock):
        index_articles_mock.return_value = [{"article_id": 3}, {"article_id": 7}]

        result = index_all_articles()

        load_ids.assert_called_once_with(20)
        index_articles_mock.assert_called_once_with([3, 7], device="cuda")
        self.assertEqual([{"article_id": 3}, {"article_id": 7}], result)

    def test_limit은_1_이상이어야_한다(self):
        with self.assertRaises(ValueError):
            index_all_articles(limit=0)


class IndexEventsTest(unittest.TestCase):
    @mock.patch("rag.indexer.db.replace_article_events")
    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses", return_value={})
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_주입된_extractor를_재사용해_model을_다시_load하지_않는다(
        self,
        load_articles,
        _load_statuses,
        load_model,
        replace_events,
    ):
        article = _article_with_chunks(3)
        load_articles.return_value = [article]
        extractor = mock.Mock()
        extractor.extract.return_value = []

        result = index_events([3], device="cpu", extractor=extractor)

        self.assertEqual("completed", result[0]["status"])
        load_model.assert_not_called()
        extractor.extract.assert_called_once_with(article["chunks"][0]["chunk_text"])
        replace_events.assert_called_once()

    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses")
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_chunk가_없는_ID를_model_load_전에_거부한다(
        self, load_articles, load_statuses, load_model
    ):
        article = _article_with_chunks(3)
        article["chunks"] = []
        load_articles.return_value = [article]

        with self.assertRaisesRegex(ValueError, "article_chunks"):
            index_events([3])

        load_statuses.assert_not_called()
        load_model.assert_not_called()

    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses")
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_같은_입력과_version의_completed_기사를_skip한다(
        self, load_articles, load_statuses, load_model
    ):
        article = _article_with_chunks(3)
        load_articles.return_value = [article]
        load_statuses.return_value = _completed_status(
            3, _source_fingerprint(article["chunks"]), event_count=2
        )

        result = index_events([3])

        self.assertEqual("skipped", result[0]["status"])
        self.assertEqual(2, result[0]["event_count"])
        load_model.assert_not_called()

    @mock.patch("rag.indexer.db.replace_article_events")
    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses", return_value={})
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_Event가_0개여도_completed로_저장한다(
        self, load_articles, _load_statuses, load_model, replace_events
    ):
        article = _article_with_chunks(3)
        load_articles.return_value = [article]
        load_model.return_value.extract.return_value = []

        result = index_events([3], device="cpu")

        self.assertEqual("completed", result[0]["status"])
        self.assertEqual(0, result[0]["event_count"])
        replace_events.assert_called_once_with(
            3,
            _source_fingerprint(article["chunks"]),
            [],
        )
        load_model.assert_called_once_with("cpu")

    @mock.patch("rag.indexer.db.replace_article_events")
    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses")
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_extraction_version이_다르면_재처리한다(
        self, load_articles, load_statuses, load_model, replace_events
    ):
        article = _article_with_chunks(3)
        load_articles.return_value = [article]
        load_statuses.return_value = _completed_status(
            3,
            _source_fingerprint(article["chunks"]),
            extraction_version="older-version",
        )
        load_model.return_value.extract.return_value = []

        result = index_events([3])

        self.assertEqual("completed", result[0]["status"])
        replace_events.assert_called_once()

    @mock.patch("rag.indexer.db.replace_article_events")
    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses", return_value={})
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_여러_parent_chunk의_같은_Event는_최고점_하나만_저장한다(
        self, load_articles, _load_statuses, load_model, replace_events
    ):
        article = _article_with_chunks(3, chunk_id=30)
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

        index_events([3])

        stored_events = replace_events.call_args.args[2]
        self.assertEqual(1, len(stored_events))
        self.assertEqual(30, stored_events[0]["source_chunk_id"])
        self.assertAlmostEqual(0.91, stored_events[0]["confidence"])

    @mock.patch("rag.indexer.db.mark_event_failed")
    @mock.patch("rag.indexer.db.replace_article_events")
    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses", return_value={})
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_기사별_실패를_기록하고_다음_기사를_계속한다(
        self,
        load_articles,
        _load_statuses,
        load_model,
        replace_events,
        mark_failed,
    ):
        first = _article_with_chunks(3, chunk_id=30)
        second = _article_with_chunks(7, chunk_id=70)
        load_articles.return_value = [first, second]
        load_model.return_value.extract.side_effect = [RuntimeError("model error"), []]

        result = index_events([3, 7])

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

    @mock.patch("rag.indexer.db.replace_article_events")
    @mock.patch("rag.indexer.load_event_extractor")
    @mock.patch("rag.indexer.db.load_event_statuses")
    @mock.patch("rag.indexer.db.load_articles_with_chunks")
    def test_force는_completed_기사도_재처리한다(
        self, load_articles, load_statuses, load_model, replace_events
    ):
        article = _article_with_chunks(3)
        load_articles.return_value = [article]
        load_statuses.return_value = _completed_status(
            3, _source_fingerprint(article["chunks"])
        )
        load_model.return_value.extract.return_value = []

        result = index_events([3], force=True)

        self.assertEqual("completed", result[0]["status"])
        replace_events.assert_called_once()


if __name__ == "__main__":
    unittest.main()
