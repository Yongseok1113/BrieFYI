"""실제 Anthropic 호출 없이 4-Layer metadata indexing 계약을 검증한다."""
import unittest
from unittest import mock

from pgvector import Vector
from pgvector.psycopg import register_vector

from db.db import get_conn
from rag.topic_indexer import index_article_topics, save_article_topics
from tests.dbhelpers import TEST_URL_PREFIX, DbTestCase, requires_db
from tools.topic_extract import extract_article_topics


class TopicExtractTest(unittest.TestCase):
    @mock.patch(
        "tools.topic_extract.call_llm",
        return_value='{"entities": [" NVIDIA ", "OpenAI", "NVIDIA"], "events": [" 투자 "]}',
    )
    def test_JSON_응답을_정리한다(self, call_llm):
        result = extract_article_topics("AI 투자 기사", None)

        self.assertEqual(["NVIDIA", "OpenAI"], result["entities"])
        self.assertEqual(["투자"], result["events"])
        self.assertEqual(500, call_llm.call_args.kwargs["max_tokens"])

    @mock.patch(
        "tools.topic_extract.call_llm",
        return_value='```json\n{"entities": [], "events": ["제품 출시"]}\n```',
    )
    def test_JSON_코드블록을_파싱한다(self, _call_llm):
        result = extract_article_topics("신제품 기사", "제품을 출시했다.")

        self.assertEqual({"entities": [], "events": ["제품 출시"]}, result)

    def test_잘못된_응답_구조를_거부한다(self):
        responses = (
            '["NVIDIA"]',
            '{"entities": "NVIDIA", "events": []}',
            '{"entities": [], "events": [1]}',
            '{"entities": ["A", "B", "C", "D"], "events": []}',
            '{"entities": [], "events": ["A", "B", "C"]}',
        )
        for response in responses:
            with self.subTest(response=response), mock.patch(
                "tools.topic_extract.call_llm", return_value=response
            ):
                with self.assertRaises(ValueError):
                    extract_article_topics("기사", "설명")


class TopicIndexerTest(unittest.TestCase):
    @mock.patch("rag.topic_indexer.save_article_topics")
    @mock.patch("rag.topic_indexer.extract_article_topics")
    @mock.patch("rag.topic_indexer.get_conn")
    def test_명시한_기사만_중복없이_처리한다(
        self, get_conn, extract_article_topics_mock, save_article_topics_mock
    ):
        conn = get_conn.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = [
            {"id": 3, "title": "세 번째 기사", "description": None},
            {"id": 7, "title": "일곱 번째 기사", "description": "설명"},
        ]
        extract_article_topics_mock.return_value = {
            "entities": ["OpenAI"],
            "events": ["제품 출시"],
        }

        results = index_article_topics([7, 3, 7], " 기술 ", [" AI ", "AI"])

        self.assertEqual([3, 7], [result["article_id"] for result in results])
        self.assertEqual(([7, 3],), conn.execute.call_args.args[1])
        self.assertEqual(2, extract_article_topics_mock.call_count)
        self.assertEqual(2, save_article_topics_mock.call_count)
        self.assertEqual("기술", results[0]["category"])
        self.assertEqual(["AI"], results[0]["domains"])

    @mock.patch("rag.topic_indexer.extract_article_topics")
    @mock.patch("rag.topic_indexer.get_conn")
    def test_존재하지_않는_article_ID를_거부한다(self, get_conn, extract_article_topics_mock):
        conn = get_conn.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = [
            {"id": 3, "title": "기사", "description": "설명"},
        ]

        with self.assertRaisesRegex(ValueError, r"\[9\]"):
            index_article_topics([3, 9], "기술", ["AI"])

        extract_article_topics_mock.assert_not_called()

    @mock.patch("rag.topic_indexer.get_conn")
    def test_UPSERT는_네_metadata_컬럼만_갱신한다(self, get_conn):
        conn = get_conn.return_value.__enter__.return_value

        save_article_topics(19, "기술", ["AI"], ["Anthropic"], ["제품 출시"])

        sql, params = conn.execute.call_args.args
        update_clause = sql.split("DO UPDATE SET", maxsplit=1)[1]
        self.assertNotIn("topic_text", update_clause)
        self.assertNotIn("embedding", update_clause)
        self.assertEqual(
            (19, "기술", ["AI"], ["Anthropic"], ["제품 출시"]),
            params,
        )


@requires_db
class TopicIndexerDbTest(DbTestCase):
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

        save_article_topics(article_id, "기술", ["AI"], ["OpenAI"], ["투자"])
        with get_conn() as conn:
            register_vector(conn)
            conn.execute(
                """UPDATE article_topics
                   SET topic_text = %s, embedding = %s
                   WHERE article_id = %s""",
                ("기술 | AI | OpenAI | 투자", Vector([0.0] * 1024), article_id),
            )

        save_article_topics(
            article_id,
            "경제",
            ["반도체"],
            ["NVIDIA"],
            ["실적 발표"],
        )
        with get_conn() as conn:
            row = conn.execute(
                """SELECT category, domains, entities, events, topic_text,
                          vector_dims(embedding) AS embedding_dimension
                   FROM article_topics
                   WHERE article_id = %s""",
                (article_id,),
            ).fetchone()

        self.assertEqual("경제", row["category"])
        self.assertEqual(["반도체"], row["domains"])
        self.assertEqual(["NVIDIA"], row["entities"])
        self.assertEqual(["실적 발표"], row["events"])
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
