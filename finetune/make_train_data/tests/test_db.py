import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unittest

from db.db import get_conn, init_db
from make_train_data.db import fetch_articles


class FetchArticlesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"DB 연결 불가: {exc}")

    def setUp(self):
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO raw_articles (digest_date, title, description, url, source, published_at)
                   VALUES (CURRENT_DATE, 'MTD 테스트 기사', '설명', 'https://test.invalid/mtd-1', 'test', now())
                   RETURNING id"""
            )
            self.article_id = cur.fetchone()["id"]

    def tearDown(self):
        with get_conn() as conn:
            conn.execute("DELETE FROM raw_articles WHERE url LIKE 'https://test.invalid/mtd-%%'")

    def test_enrichment_없어도_기사가_조회된다(self):
        rows = fetch_articles()
        matched = [r for r in rows if r["id"] == self.article_id]
        self.assertEqual(len(matched), 1)
        self.assertIsNone(matched[0]["entity"])

    def test_enrichment_있으면_함께_조회된다(self):
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO enrichment (raw_article_id, insights, category, domain, entity, event)
                   VALUES (%s, '[]', '기술', '["AI"]', '["NVIDIA"]', '["출시"]')""",
                (self.article_id,),
            )
        rows = fetch_articles()
        matched = [r for r in rows if r["id"] == self.article_id]
        self.assertEqual(matched[0]["category"], "기술")
        self.assertEqual(matched[0]["entity"], ["NVIDIA"])


if __name__ == "__main__":
    unittest.main()
