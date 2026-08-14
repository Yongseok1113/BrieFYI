"""테이블별 CRUD 테스트 (raw_articles / digests / send_log).

각 테이블에 대해 INSERT → SELECT → UPDATE → DELETE를 직접 SQL로 확인하고,
제약조건(UNIQUE, NOT NULL, FK CASCADE)과 트랜잭션 동작도 함께 검증한다.
DB가 없으면 skip된다.

    python -m unittest tests.test_db_crud
"""
import unittest

import psycopg
from psycopg.types.json import Jsonb

from db.db import get_conn
from tests.dbhelpers import TEST_KEYWORD, TEST_URL_PREFIX, DbTestCase, requires_db


@requires_db
class RawArticlesCrudTest(DbTestCase):
    def _insert(self, n: int, **overrides) -> int:
        a = self.article(n, **overrides)
        with get_conn() as conn:
            row = conn.execute(
                """INSERT INTO raw_articles
                   (digest_date, title, description, url, source, published_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    self.today,
                    a["title"],
                    a["description"],
                    a["url"],
                    a["source"],
                    a["published_at"] or None,
                ),
            ).fetchone()
        return row["id"]

    def test_생성(self):
        article_id = self._insert(1)

        self.assertIsInstance(article_id, int)
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM raw_articles WHERE id = %s", (article_id,)).fetchone()

        self.assertEqual("테스트 기사 1", row["title"])
        self.assertEqual(TEST_URL_PREFIX + "1", row["url"])
        self.assertIsNotNone(row["fetched_at"], "fetched_at은 DEFAULT now()로 채워진다")

    def test_조회(self):
        self._insert(1)
        self._insert(2)

        with get_conn() as conn:
            rows = conn.execute(
                """SELECT title, url FROM raw_articles
                   WHERE digest_date = %s AND url LIKE %s
                   ORDER BY url""",
                (self.today, TEST_URL_PREFIX + "%"),
            ).fetchall()

        self.assertEqual([TEST_URL_PREFIX + "1", TEST_URL_PREFIX + "2"], [r["url"] for r in rows])

    def test_수정(self):
        article_id = self._insert(1)

        with get_conn() as conn:
            cur = conn.execute(
                "UPDATE raw_articles SET title = %s, source = %s WHERE id = %s",
                ("수정된 제목", "updated", article_id),
            )
            self.assertEqual(1, cur.rowcount)

        with get_conn() as conn:
            row = conn.execute(
                "SELECT title, source FROM raw_articles WHERE id = %s", (article_id,)
            ).fetchone()

        self.assertEqual("수정된 제목", row["title"])
        self.assertEqual("updated", row["source"])

    def test_삭제(self):
        article_id = self._insert(1)

        with get_conn() as conn:
            cur = conn.execute("DELETE FROM raw_articles WHERE id = %s", (article_id,))
            self.assertEqual(1, cur.rowcount)

        with get_conn() as conn:
            self.assertIsNone(
                conn.execute("SELECT id FROM raw_articles WHERE id = %s", (article_id,)).fetchone()
            )

    def test_url이_중복되면_UNIQUE_제약에_걸린다(self):
        self._insert(1)

        with self.assertRaises(psycopg.errors.UniqueViolation):
            self._insert(1)

    def test_title이_없으면_NOT_NULL_제약에_걸린다(self):
        with self.assertRaises(psycopg.errors.NotNullViolation):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO raw_articles (digest_date, title, url) VALUES (%s, %s, %s)",
                    (self.today, None, TEST_URL_PREFIX + "no-title"),
                )


@requires_db
class DigestsCrudTest(DbTestCase):
    SUMMARY = [{"title": "기사", "summary": "요약"}]
    INSIGHT = {"insights": ["인사이트1", "인사이트2"], "implication": "시사점"}

    def _insert(self) -> int:
        with get_conn() as conn:
            row = conn.execute(
                """INSERT INTO digests (digest_date, keyword, summary_json, insight_json)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (self.today, TEST_KEYWORD, Jsonb(self.SUMMARY), Jsonb(self.INSIGHT)),
            ).fetchone()
        return row["id"]

    def test_생성과_조회(self):
        digest_id = self._insert()

        with get_conn() as conn:
            row = conn.execute("SELECT * FROM digests WHERE id = %s", (digest_id,)).fetchone()

        # jsonb는 파이썬 객체로 그대로 돌아온다.
        self.assertEqual(self.SUMMARY, row["summary_json"])
        self.assertEqual(self.INSIGHT, row["insight_json"])
        self.assertIsNotNone(row["created_at"])

    def test_jsonb_내부_값으로_조회할_수_있다(self):
        self._insert()

        with get_conn() as conn:
            row = conn.execute(
                """SELECT jsonb_array_length(summary_json) AS n,
                          insight_json->>'implication' AS implication
                   FROM digests WHERE keyword = %s""",
                (TEST_KEYWORD,),
            ).fetchone()

        self.assertEqual(1, row["n"])
        self.assertEqual("시사점", row["implication"])

    def test_수정(self):
        digest_id = self._insert()
        new_insight = {"insights": ["교체됨"], "implication": "새 시사점"}

        with get_conn() as conn:
            cur = conn.execute(
                "UPDATE digests SET insight_json = %s WHERE id = %s", (Jsonb(new_insight), digest_id)
            )
            self.assertEqual(1, cur.rowcount)

        with get_conn() as conn:
            row = conn.execute(
                "SELECT insight_json FROM digests WHERE id = %s", (digest_id,)
            ).fetchone()

        self.assertEqual(new_insight, row["insight_json"])

    def test_jsonb_일부만_수정한다(self):
        digest_id = self._insert()

        with get_conn() as conn:
            conn.execute(
                """UPDATE digests
                   SET insight_json = jsonb_set(insight_json, '{implication}', %s::jsonb)
                   WHERE id = %s""",
                ('"부분 수정된 시사점"', digest_id),
            )

        with get_conn() as conn:
            row = conn.execute(
                "SELECT insight_json FROM digests WHERE id = %s", (digest_id,)
            ).fetchone()

        self.assertEqual("부분 수정된 시사점", row["insight_json"]["implication"])
        self.assertEqual(self.INSIGHT["insights"], row["insight_json"]["insights"], "다른 키는 유지된다")

    def test_삭제(self):
        digest_id = self._insert()

        with get_conn() as conn:
            cur = conn.execute("DELETE FROM digests WHERE id = %s", (digest_id,))
            self.assertEqual(1, cur.rowcount)

        with get_conn() as conn:
            self.assertIsNone(
                conn.execute("SELECT id FROM digests WHERE id = %s", (digest_id,)).fetchone()
            )


@requires_db
class SendLogCrudTest(DbTestCase):
    def setUp(self):
        super().setUp()
        with get_conn() as conn:
            self.digest_id = conn.execute(
                """INSERT INTO digests (digest_date, keyword, summary_json, insight_json)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (self.today, TEST_KEYWORD, Jsonb([]), Jsonb({})),
            ).fetchone()["id"]

    def _insert(self, status: str = "success", error: str | None = None) -> int:
        with get_conn() as conn:
            return conn.execute(
                """INSERT INTO send_log (digest_id, channel, recipient, status, error)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (self.digest_id, "email", "to@example.com", status, error),
            ).fetchone()["id"]

    def test_생성과_조회(self):
        log_id = self._insert()

        with get_conn() as conn:
            row = conn.execute("SELECT * FROM send_log WHERE id = %s", (log_id,)).fetchone()

        self.assertEqual(self.digest_id, row["digest_id"])
        self.assertEqual("email", row["channel"])
        self.assertEqual("success", row["status"])
        self.assertIsNone(row["error"])
        self.assertIsNotNone(row["sent_at"])

    def test_digest와_조인해서_조회한다(self):
        self._insert(status="failed", error="발송 실패")

        with get_conn() as conn:
            row = conn.execute(
                """SELECT d.keyword, s.status, s.error
                   FROM digests d JOIN send_log s ON s.digest_id = d.id
                   WHERE d.id = %s""",
                (self.digest_id,),
            ).fetchone()

        self.assertEqual(TEST_KEYWORD, row["keyword"])
        self.assertEqual("failed", row["status"])
        self.assertEqual("발송 실패", row["error"])

    def test_수정(self):
        log_id = self._insert(status="failed", error="타임아웃")

        with get_conn() as conn:
            conn.execute(
                "UPDATE send_log SET status = %s, error = NULL WHERE id = %s", ("success", log_id)
            )

        with get_conn() as conn:
            row = conn.execute(
                "SELECT status, error FROM send_log WHERE id = %s", (log_id,)
            ).fetchone()

        self.assertEqual("success", row["status"])
        self.assertIsNone(row["error"])

    def test_삭제(self):
        log_id = self._insert()

        with get_conn() as conn:
            cur = conn.execute("DELETE FROM send_log WHERE id = %s", (log_id,))
            self.assertEqual(1, cur.rowcount)

        with get_conn() as conn:
            self.assertIsNone(
                conn.execute("SELECT id FROM send_log WHERE id = %s", (log_id,)).fetchone()
            )

    def test_없는_digest_id는_FK_제약에_걸린다(self):
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            with get_conn() as conn:
                conn.execute(
                    """INSERT INTO send_log (digest_id, channel, status)
                       VALUES (%s, %s, %s)""",
                    (-1, "email", "success"),
                )

    def test_digest를_지우면_발송이력도_함께_지워진다(self):
        log_id = self._insert()

        with get_conn() as conn:
            conn.execute("DELETE FROM digests WHERE id = %s", (self.digest_id,))

        with get_conn() as conn:
            self.assertIsNone(
                conn.execute("SELECT id FROM send_log WHERE id = %s", (log_id,)).fetchone(),
                "ON DELETE CASCADE로 함께 삭제되어야 한다",
            )


@requires_db
class TransactionTest(DbTestCase):
    """get_conn()의 커밋/롤백 경계."""

    def test_한_컨텍스트의_여러_변경이_함께_커밋된다(self):
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO raw_articles (digest_date, title, url) VALUES (%s, %s, %s)",
                (self.today, "커밋1", TEST_URL_PREFIX + "commit-1"),
            )
            conn.execute(
                "INSERT INTO raw_articles (digest_date, title, url) VALUES (%s, %s, %s)",
                (self.today, "커밋2", TEST_URL_PREFIX + "commit-2"),
            )

        with get_conn() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM raw_articles WHERE url LIKE %s", (TEST_URL_PREFIX + "commit-%",)
            ).fetchone()

        self.assertEqual(2, row["n"])

    def test_예외가_나면_같은_컨텍스트의_변경이_모두_롤백된다(self):
        with self.assertRaises(psycopg.errors.UniqueViolation):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO raw_articles (digest_date, title, url) VALUES (%s, %s, %s)",
                    (self.today, "롤백 대상", TEST_URL_PREFIX + "rollback-1"),
                )
                # 같은 URL을 두 번 넣어 UNIQUE 위반을 유발한다.
                conn.execute(
                    "INSERT INTO raw_articles (digest_date, title, url) VALUES (%s, %s, %s)",
                    (self.today, "중복", TEST_URL_PREFIX + "rollback-1"),
                )

        with get_conn() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM raw_articles WHERE url LIKE %s", (TEST_URL_PREFIX + "rollback-%",)
            ).fetchone()

        self.assertEqual(0, row["n"], "첫 INSERT까지 롤백되어야 한다")


if __name__ == "__main__":
    unittest.main()
