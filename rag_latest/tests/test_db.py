"""rag/db.py의 ID 계약과, 실제 PostgreSQL에 대한 저장·검색 SQL을 검증한다.

@requires_db 클래스는 pgvector가 설치된 PostgreSQL이 떠 있을 때만 실행된다
(scripts/db-up.ps1 또는 ./scripts/db-up.sh).
"""
import unittest
from unittest import mock

from pgvector import Vector
from pgvector.psycopg import register_vector

from db.db import get_conn
from rag_latest import db
from rag_latest.retriever import retrieve
from tests.dbhelpers import TEST_URL_PREFIX, DbTestCase, requires_db


class ArticleIdsTest(unittest.TestCase):
    def test_요청_ID를_순서대로_중복없이_정리한다(self):
        self.assertEqual([7, 3], db.normalize_article_ids([7, 3, 7, "3"]))

    def test_존재하지_않는_ID를_거부한다(self):
        with self.assertRaisesRegex(ValueError, r"\[9\]"):
            db._require_all_found([3, 9], {3})

    @mock.patch("rag_latest.db.get_conn")
    def test_UPSERT는_Category_Domain_Entity만_갱신한다(self, get_conn_mock):
        conn = get_conn_mock.return_value.__enter__.return_value

        db.save_article_topics(19, "기술", ["AI"], ["Anthropic"])

        sql, params = conn.execute.call_args.args
        update_clause = sql.split("DO UPDATE SET", maxsplit=1)[1]
        self.assertNotIn("topic_text", update_clause)
        self.assertNotIn("embedding", update_clause)
        self.assertEqual((19, "기술", ["AI"], ["Anthropic"]), params)


@requires_db
class RagSearchDbTest(DbTestCase):
    """고정 3차원 vector로 실제 PostgreSQL 검색 SQL을 검증한다."""

    MODEL = "__test_embedding_model__"

    def setUp(self):
        super().setUp()
        with get_conn() as conn:
            register_vector(conn)
            self.gym_chunk_id = self._insert_chunk(
                conn,
                suffix="rag-gym",
                title="AI agent gym hacking incident",
                description="An autonomous agent accessed a gym booking system.",
                vector=[1.0, 0.0, 0.0],
            )
            self.stock_chunk_id = self._insert_chunk(
                conn,
                suffix="rag-stock",
                title="Stock market valuation risk",
                description="Investors discuss expensive AI stocks.",
                vector=[0.0, 1.0, 0.0],
            )
            self.gym_article_id = conn.execute(
                "SELECT article_id FROM article_chunks WHERE id = %s",
                (self.gym_chunk_id,),
            ).fetchone()["article_id"]
            self.stock_article_id = conn.execute(
                "SELECT article_id FROM article_chunks WHERE id = %s",
                (self.stock_chunk_id,),
            ).fetchone()["article_id"]
        db.save_article_topics(self.gym_article_id, "기술", ["AI"], ["Agent"])
        db.save_article_topics(self.stock_article_id, "경제", ["증시"], ["Investor"])

    def _insert_chunk(self, conn, suffix, title, description, vector):
        article_id = conn.execute(
            """INSERT INTO raw_articles (digest_date, title, description, url, source)
               VALUES (%s, %s, %s, %s, 'test') RETURNING id""",
            (self.today, title, description, TEST_URL_PREFIX + suffix),
        ).fetchone()["id"]
        chunk_id = conn.execute(
            """INSERT INTO article_chunks (article_id, chunk_index, chunk_text)
               VALUES (%s, 0, %s) RETURNING id""",
            (article_id, f"{title}\n\n{description}"),
        ).fetchone()["id"]
        conn.execute(
            """INSERT INTO chunk_embeddings
                   (chunk_id, embedding_model, embedding_dimension, embedding)
               VALUES (%s, %s, 3, %s)""",
            (chunk_id, self.MODEL, Vector(vector)),
        )
        return chunk_id

    @mock.patch("rag_latest.db.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag_latest.db.config.HF_EMBEDDING_MODEL", MODEL)
    @mock.patch("rag_latest.retriever.embed_query", return_value=[1.0, 0.0, 0.0])
    def test_cosine_vector_search(self, _embed_query):
        rows = retrieve("semantic query", top_k=2, search_mode="vector")

        self.assertEqual(self.gym_chunk_id, rows[0]["chunk_id"])
        self.assertAlmostEqual(1.0, rows[0]["score"])

    def test_text_search(self):
        rows = retrieve("gym hacking", top_k=2, search_mode="text")

        self.assertEqual([self.gym_chunk_id], [row["chunk_id"] for row in rows])
        self.assertGreater(rows[0]["score"], 0)

    @mock.patch("rag_latest.db.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag_latest.db.config.HF_EMBEDDING_MODEL", MODEL)
    @mock.patch("rag_latest.retriever.embed_query", return_value=[1.0, 0.0, 0.0])
    def test_rrf_hybrid_search(self, _embed_query):
        rows = retrieve(
            "stock market",
            top_k=2,
            search_mode="hybrid",
            vector_weight=0.7,
            text_weight=0.3,
        )

        self.assertEqual(self.stock_chunk_id, rows[0]["chunk_id"])
        self.assertEqual(self.gym_chunk_id, rows[1]["chunk_id"])
        self.assertEqual(1, rows[0]["text_rank"])
        self.assertGreater(rows[0]["score"], rows[1]["score"])

    @mock.patch("rag_latest.db.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag_latest.db.config.HF_EMBEDDING_MODEL", MODEL)
    @mock.patch("rag_latest.retriever.embed_query", return_value=[0.71, 0.70, 0.0])
    def test_Category_Domain_가산점이_비슷한_vector_후보를_재정렬한다(self, _embed_query):
        rows = retrieve(
            "market AI",
            top_k=2,
            search_mode="vector",
            category="경제",
            domains=["증시"],
        )

        self.assertEqual(self.stock_chunk_id, rows[0]["chunk_id"])
        self.assertTrue(rows[0]["category_match"])
        self.assertEqual(["증시"], rows[0]["matched_domains"])
        self.assertGreater(rows[0]["metadata_score"], 0)


@requires_db
class ArticleIndexDbTest(DbTestCase):
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

        selected_ids = db.load_unindexed_article_ids(100000)

        self.assertIn(missing_id, selected_ids)
        self.assertNotIn(indexed_id, selected_ids)

    @mock.patch("rag_latest.db.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag_latest.db.config.HF_EMBEDDING_MODEL", "__test_embedding_model__")
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

        db.store_article_index(
            article_id,
            [
                {"chunk_index": 0, "chunk_text": "새 본문 첫 번째 chunk"},
                {"chunk_index": 1, "chunk_text": "새 본문 두 번째 chunk"},
            ],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
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

        self.assertEqual(2, counts["chunks"])
        self.assertEqual(2, counts["embeddings"])
        self.assertEqual(0, counts["events"])
        self.assertEqual(0, counts["statuses"])
        self.assertEqual(0, counts["arguments"])


@requires_db
class EventStoreDbTest(DbTestCase):
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

        db.replace_article_events(
            article_id,
            "source-fingerprint",
            [
                {
                    "event_type": "invested_in",
                    "confidence": 0.9,
                    "event_fingerprint": "event-fingerprint",
                    "evidence_start": 0,
                    "evidence_end": 18,
                    "source_chunk_id": chunk_id,
                    "arguments": [
                        {
                            "role": "investor",
                            "text": "NVIDIA가",
                            "normalized_text": "NVIDIA",
                            "confidence": 0.9,
                            "span_start": 0,
                            "span_end": 7,
                        },
                        {
                            "role": "investee",
                            "text": "OpenAI에",
                            "normalized_text": "OpenAI",
                            "confidence": 0.9,
                            "span_start": 8,
                            "span_end": 15,
                        },
                    ],
                }
            ],
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


@requires_db
class TopicStoreDbTest(DbTestCase):
    def test_UPSERT와_원문삭제_cascade를_DB에서_검증한다(self):
        with get_conn() as conn:
            article_id = conn.execute(
                """INSERT INTO raw_articles (digest_date, title, description, url)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (
                    self.today,
                    "4-Layer 테스트 기사",
                    "테스트 설명",
                    TEST_URL_PREFIX + "topic-indexer",
                ),
            ).fetchone()["id"]

        db.save_article_topics(article_id, "기술", ["AI"], ["OpenAI"])
        with get_conn() as conn:
            register_vector(conn)
            conn.execute(
                """UPDATE article_topics
                   SET topic_text = %s, embedding = %s
                   WHERE article_id = %s""",
                ("기술 | AI | OpenAI | 투자", Vector([0.0] * 1024), article_id),
            )

        db.save_article_topics(article_id, "경제", ["반도체"], ["NVIDIA"])
        with get_conn() as conn:
            row = conn.execute(
                """SELECT category, domains, entities, topic_text,
                          vector_dims(embedding) AS embedding_dimension
                   FROM article_topics
                   WHERE article_id = %s""",
                (article_id,),
            ).fetchone()

        self.assertEqual("경제", row["category"])
        self.assertEqual(["반도체"], row["domains"])
        self.assertEqual(["NVIDIA"], row["entities"])
        self.assertEqual("기술 | AI | OpenAI | 투자", row["topic_text"])
        self.assertEqual(1024, row["embedding_dimension"])

        with get_conn() as conn:
            conn.execute("DELETE FROM raw_articles WHERE id = %s", (article_id,))
            remaining = conn.execute(
                "SELECT count(*) AS count FROM article_topics WHERE article_id = %s",
                (article_id,),
            ).fetchone()["count"]
        self.assertEqual(0, remaining)


if __name__ == "__main__":
    unittest.main()
