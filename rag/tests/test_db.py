"""고정 3차원 vector로 실제 PostgreSQL 검색 SQL을 검증한다."""
import unittest
from unittest import mock

from pgvector import Vector
from pgvector.psycopg import register_vector

from db.db import get_conn
from rag.retriever import retrieve
from tests.dbhelpers import TEST_URL_PREFIX, DbTestCase, requires_db


@requires_db
class RagSearchDbTest(DbTestCase):
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

    @mock.patch("rag.retriever.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag.retriever.config.HF_EMBEDDING_MODEL", MODEL)
    @mock.patch("rag.retriever.embed_query", return_value=[1.0, 0.0, 0.0])
    def test_cosine_vector_search(self, _embed_query):
        rows = retrieve("semantic query", top_k=2, search_mode="vector", metric="cosine")

        self.assertEqual(self.gym_chunk_id, rows[0]["chunk_id"])
        self.assertAlmostEqual(1.0, rows[0]["score"])

    def test_text_search(self):
        rows = retrieve("gym hacking", top_k=2, search_mode="text")

        self.assertEqual([self.gym_chunk_id], [row["chunk_id"] for row in rows])
        self.assertGreater(rows[0]["score"], 0)

    @mock.patch("rag.retriever.config.HF_EMBEDDING_DIMENSION", 3)
    @mock.patch("rag.retriever.config.HF_EMBEDDING_MODEL", MODEL)
    @mock.patch("rag.retriever.embed_query", return_value=[1.0, 0.0, 0.0])
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


if __name__ == "__main__":
    unittest.main()
