"""실제 DB와 HF 호출 없이 indexer의 기사 선택 계약을 검증한다."""
import unittest
from unittest import mock

from pgvector import Vector
from pgvector.psycopg import register_vector

from db.db import get_conn
from rag.indexer import index_all_articles
from tests.dbhelpers import TEST_URL_PREFIX, DbTestCase, requires_db


class IndexAllArticlesTest(unittest.TestCase):
    @mock.patch("rag.indexer.index_articles")
    @mock.patch("rag.indexer.get_conn")
    def test_embedding이_없는_기사만_넘긴다(self, get_conn, index_articles):
        conn = get_conn.return_value.__enter__.return_value
        conn.execute.return_value = [{"id": 3}, {"id": 7}]
        index_articles.return_value = [{"article_id": 3}, {"article_id": 7}]

        result = index_all_articles()

        index_articles.assert_called_once_with([3, 7])
        self.assertEqual([{"article_id": 3}, {"article_id": 7}], result)

        (sql,) = conn.execute.call_args.args
        self.assertIn("NOT EXISTS", sql)
        self.assertIn("chunk_embeddings", sql)


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
            index_all_articles()

        (selected_ids,) = index_articles.call_args.args
        self.assertIn(missing_id, selected_ids)
        self.assertNotIn(indexed_id, selected_ids)


if __name__ == "__main__":
    unittest.main()
