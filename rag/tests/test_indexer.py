"""실제 DB와 HF 호출 없이 indexer의 기사 선택 계약을 검증한다."""
import unittest
from unittest import mock

from pgvector import Vector
from pgvector.psycopg import register_vector

from db.db import get_conn
from rag.indexer import _store_article, index_all_articles, index_articles
from tests.dbhelpers import TEST_URL_PREFIX, DbTestCase, requires_db
from tools.article_content import ArticleContentDependencyError


class OffsetTokenizer:
    def __call__(self, text, **kwargs):
        offsets = []
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"offset_mapping": offsets}


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

    @staticmethod
    def _event_extractor():
        extractor = mock.Mock()
        extractor.model = mock.Mock()
        return extractor

    def test_본문_수집에_성공하면_본문을_청킹한다(self):
        with (
            mock.patch("rag.indexer._load_articles", return_value=[self.ARTICLE]),
            mock.patch("rag.indexer.load_embedding_tokenizer", return_value=OffsetTokenizer()),
            mock.patch(
                "rag.indexer.load_event_extractor",
                return_value=self._event_extractor(),
            ),
            mock.patch("rag.indexer.GLiNER2TopicExtractor") as topic_extractor,
            mock.patch("rag.indexer.fetch_article_body", return_value="실제 기사 본문"),
            mock.patch("rag.indexer.embed_texts", return_value=[[1.0, 0.0, 0.0]]) as embed,
            mock.patch("rag.indexer._store_article", return_value={"article_id": 3}) as store,
            mock.patch(
                "rag.indexer.index_event_articles",
                return_value=[{"article_id": 3, "status": "completed", "event_count": 1}],
            ),
        ):
            topic_extractor.return_value.extract.return_value = self.TOPICS
            result = index_articles([3])

        embed.assert_called_once_with(["기사 제목\n\n실제 기사 본문"])
        self.assertEqual("body", store.call_args.args[3])
        self.assertEqual(self.TOPICS, store.call_args.args[4])
        self.assertEqual("completed", result[0]["event_status"])

    def test_본문_수집에_실패하면_title_description을_사용한다(self):
        with (
            mock.patch("rag.indexer._load_articles", return_value=[self.ARTICLE]),
            mock.patch("rag.indexer.load_embedding_tokenizer", return_value=OffsetTokenizer()),
            mock.patch(
                "rag.indexer.load_event_extractor",
                return_value=self._event_extractor(),
            ),
            mock.patch("rag.indexer.GLiNER2TopicExtractor") as topic_extractor,
            mock.patch("rag.indexer.fetch_article_body", side_effect=RuntimeError("blocked")),
            mock.patch("rag.indexer.embed_texts", return_value=[[1.0, 0.0, 0.0]]) as embed,
            mock.patch("rag.indexer._store_article", return_value={"article_id": 3}) as store,
            mock.patch(
                "rag.indexer.index_event_articles",
                return_value=[{"article_id": 3, "status": "completed", "event_count": 0}],
            ),
        ):
            topic_extractor.return_value.extract.return_value = self.TOPICS
            index_articles([3])

        embed.assert_called_once_with(["기사 제목\n\n짧은 설명"])
        self.assertEqual("title+description", store.call_args.args[3])

    def test_본문_추출_의존성_누락은_fallback으로_숨기지_않는다(self):
        with (
            mock.patch("rag.indexer._load_articles", return_value=[self.ARTICLE]),
            mock.patch("rag.indexer.load_embedding_tokenizer", return_value=OffsetTokenizer()),
            mock.patch(
                "rag.indexer.load_event_extractor",
                return_value=self._event_extractor(),
            ),
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
    @mock.patch("rag.indexer.get_conn")
    def test_embedding이_없는_기사만_넘긴다(self, get_conn, index_articles):
        conn = get_conn.return_value.__enter__.return_value
        conn.execute.return_value = [{"id": 3}, {"id": 7}]
        index_articles.return_value = [{"article_id": 3}, {"article_id": 7}]

        result = index_all_articles()

        index_articles.assert_called_once_with([3, 7], device="cuda")
        self.assertEqual([{"article_id": 3}, {"article_id": 7}], result)

        sql, params = conn.execute.call_args.args
        self.assertIn("NOT EXISTS", sql)
        self.assertIn("chunk_embeddings", sql)
        self.assertIn("LIMIT", sql)
        self.assertEqual((20,), params)

    def test_limit은_1_이상이어야_한다(self):
        with self.assertRaises(ValueError):
            index_all_articles(limit=0)


@requires_db
class IndexAllArticlesDbTest(DbTestCase):
    def test_embedding이_없는_기사만_DB에서_선택한다(self):
        with get_conn() as conn:
            register_vector(conn)
            missing_id = conn.execute(
                """INSERT INTO raw_articles (digest_date, title, url)
                   VALUES (%s, %s, %s) RETURNING id""",
                (self.today, "미인덱싱 기사", TEST_URL_PREFIX + "indexer-missing"),
            ).fetchone()["id"]
            indexed_id = conn.execute(
                """INSERT INTO raw_articles (digest_date, title, url)
                   VALUES (%s, %s, %s) RETURNING id""",
                (self.today, "인덱싱된 기사", TEST_URL_PREFIX + "indexer-done"),
            ).fetchone()["id"]
            chunk_id = conn.execute(
                """INSERT INTO article_chunks (article_id, chunk_index, chunk_text)
                   VALUES (%s, 0, %s) RETURNING id""",
                (indexed_id, "인덱싱된 기사"),
            ).fetchone()["id"]
            conn.execute(
                """INSERT INTO chunk_embeddings
                       (chunk_id, embedding_model, embedding_dimension, embedding)
                   VALUES (%s, %s, 3, %s)""",
                (chunk_id, "__test_embedding_model__", Vector([1.0, 0.0, 0.0])),
            )

        with mock.patch("rag.indexer.index_articles", return_value=[]) as index_articles:
            index_all_articles(limit=100000)

        (selected_ids,) = index_articles.call_args.args
        self.assertIn(missing_id, selected_ids)
        self.assertNotIn(indexed_id, selected_ids)
        self.assertEqual("cuda", index_articles.call_args.kwargs["device"])

    @mock.patch("rag.indexer.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag.indexer.config.HF_EMBEDDING_MODEL", "__test_embedding_model__")
    def test_chunk가_바뀌면_embedding과_Event를_교체한다(self):
        with get_conn() as conn:
            register_vector(conn)
            article_id = conn.execute(
                """INSERT INTO raw_articles (digest_date, title, description, url)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (
                    self.today,
                    "본문 재인덱싱 기사",
                    "기존 설명",
                    TEST_URL_PREFIX + "indexer-replace",
                ),
            ).fetchone()["id"]
            old_chunk_id = conn.execute(
                """INSERT INTO article_chunks (article_id, chunk_index, chunk_text)
                   VALUES (%s, 0, %s) RETURNING id""",
                (article_id, "기존 title+description chunk"),
            ).fetchone()["id"]
            conn.execute(
                """INSERT INTO chunk_embeddings
                       (chunk_id, embedding_model, embedding_dimension, embedding)
                   VALUES (%s, %s, 3, %s)""",
                (old_chunk_id, "__test_embedding_model__", Vector([1.0, 0.0, 0.0])),
            )
            conn.execute(
                """INSERT INTO article_event_index_status
                       (article_id, extractor_model, taxonomy_version, extraction_version,
                        source_fingerprint, status, event_count)
                   VALUES (%s, 'test', 'test', 'test', 'old', 'completed', 1)""",
                (article_id,),
            )
            event_id = conn.execute(
                """INSERT INTO article_events
                       (article_id, source_chunk_id, event_type, confidence, extractor_model,
                        taxonomy_version, extraction_version, event_fingerprint,
                        evidence_start, evidence_end)
                   VALUES (%s, %s, 'released', 0.9, 'test', 'test', 'test',
                           'old-event', 0, 5)
                   RETURNING id""",
                (article_id, old_chunk_id),
            ).fetchone()["id"]
            conn.execute(
                """INSERT INTO article_event_arguments
                       (event_id, argument_index, role, text, normalized_text,
                        confidence, span_start, span_end)
                   VALUES (%s, 0, 'releaser', '기업', '기업', 0.9, 0, 2)""",
                (event_id,),
            )

        result = _store_article(
            {"id": article_id},
            [
                {"chunk_index": 0, "chunk_text": "새 본문 첫 번째 chunk"},
                {"chunk_index": 1, "chunk_text": "새 본문 두 번째 chunk"},
            ],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "body",
            {
                "category": "기술",
                "domains": ["AI"],
                "entities": ["OpenAI"],
            },
        )

        with get_conn() as conn:
            counts = conn.execute(
                """SELECT
                       (SELECT count(*) FROM article_chunks WHERE article_id = %s) AS chunks,
                       (SELECT count(*) FROM chunk_embeddings ce
                        JOIN article_chunks ac ON ac.id = ce.chunk_id
                        WHERE ac.article_id = %s) AS embeddings,
                       (SELECT count(*) FROM article_events WHERE article_id = %s) AS events,
                       (SELECT count(*) FROM article_event_index_status
                        WHERE article_id = %s) AS statuses,
                       (SELECT count(*) FROM article_event_arguments aa
                        JOIN article_events ae ON ae.id = aa.event_id
                        WHERE ae.article_id = %s) AS arguments""",
                (article_id, article_id, article_id, article_id, article_id),
            ).fetchone()

        self.assertEqual(2, result["chunk_count"])
        self.assertEqual("body", result["text_source"])
        self.assertEqual(2, counts["chunks"])
        self.assertEqual(2, counts["embeddings"])
        self.assertEqual(0, counts["events"])
        self.assertEqual(0, counts["statuses"])
        self.assertEqual(0, counts["arguments"])


if __name__ == "__main__":
    unittest.main()
