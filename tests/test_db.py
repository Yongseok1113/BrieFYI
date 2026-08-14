"""db/db.py 헬퍼 함수 테스트 (파이프라인이 실제로 호출하는 경로).

테이블별 원시 CRUD는 tests/test_db_crud.py, 접속은 tests/test_db_connection.py에 있다.
DB가 없으면 skip된다.

    python -m unittest tests.test_db
"""
import unittest
from datetime import date

import psycopg

from db.db import get_conn, insert_articles, log_send, save_digest
from tests.dbhelpers import TEST_KEYWORD, TEST_URL_PREFIX, DbTestCase, requires_db


@requires_db
class InsertArticlesTest(DbTestCase):
    def test_기사를_저장하고_중복_URL은_건너뛴다(self):
        first = insert_articles(self.today, [self.article(1), self.article(2)])
        second = insert_articles(self.today, [self.article(2), self.article(3)])

        self.assertEqual(2, first)
        self.assertEqual(1, second, "이미 저장된 URL은 세지 않아야 한다")

        with get_conn() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM raw_articles WHERE url LIKE %s", (TEST_URL_PREFIX + "%",)
            ).fetchone()
        self.assertEqual(3, row["n"])

    def test_published_at이_비어도_저장된다(self):
        article = self.article(9, published_at="")

        self.assertEqual(1, insert_articles(self.today, [article]))

        with get_conn() as conn:
            row = conn.execute(
                "SELECT published_at FROM raw_articles WHERE url = %s", (article["url"],)
            ).fetchone()
        self.assertIsNone(row["published_at"])

    def test_빈_리스트를_넣으면_0건이다(self):
        self.assertEqual(0, insert_articles(self.today, []))


@requires_db
class DigestAndSendLogTest(DbTestCase):
    def test_다이제스트와_발송이력을_저장한다(self):
        summaries = [{"title": "제목", "summary": "요약"}]
        insight = {"insights": ["인사이트1"], "implication": "시사점"}

        digest_id = save_digest(self.today, TEST_KEYWORD, summaries, insight)
        log_send(digest_id, "email", "to@example.com", "success")

        self.assertIsInstance(digest_id, int)
        with get_conn() as conn:
            digest = conn.execute("SELECT * FROM digests WHERE id = %s", (digest_id,)).fetchone()
            sent = conn.execute("SELECT * FROM send_log WHERE digest_id = %s", (digest_id,)).fetchall()

        # JSONB로 저장되므로 파이썬 객체 그대로 돌아온다.
        self.assertEqual(summaries, digest["summary_json"])
        self.assertEqual(insight, digest["insight_json"])
        self.assertEqual(1, len(sent))
        self.assertEqual("success", sent[0]["status"])

    def test_실패_이력도_사유와_함께_남는다(self):
        digest_id = save_digest(self.today, TEST_KEYWORD, [], {})

        log_send(digest_id, "email", "to@example.com", "failed", "SMTP timeout")

        with get_conn() as conn:
            row = conn.execute(
                "SELECT status, error FROM send_log WHERE digest_id = %s", (digest_id,)
            ).fetchone()
        self.assertEqual("failed", row["status"])
        self.assertEqual("SMTP timeout", row["error"])

    def test_save_digest는_매번_새_id를_반환한다(self):
        first = save_digest(self.today, TEST_KEYWORD, [], {})
        second = save_digest(self.today, TEST_KEYWORD, [], {})

        self.assertNotEqual(first, second)
        self.assertGreater(second, first)


@requires_db
class GetConnTest(DbTestCase):
    def test_저장_중_예외가_나면_롤백된다(self):
        with self.assertRaises(psycopg.Error):
            with get_conn() as conn:
                conn.execute(
                    """INSERT INTO raw_articles (digest_date, title, url)
                       VALUES (%s, %s, %s)""",
                    (date.today().isoformat(), "롤백 대상", TEST_URL_PREFIX + "rollback"),
                )
                conn.execute("SELECT 1 FROM 존재하지_않는_테이블")

        with get_conn() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM raw_articles WHERE url = %s", (TEST_URL_PREFIX + "rollback",)
            ).fetchone()
        self.assertEqual(0, row["n"], "예외 발생 시 커밋되지 않아야 한다")


if __name__ == "__main__":
    unittest.main()
