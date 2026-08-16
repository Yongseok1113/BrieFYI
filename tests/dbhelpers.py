"""DB 통합 테스트 공용 헬퍼.

실행 중인 PostgreSQL이 필요하다(`scripts/db-up.ps1` 또는 `./scripts/db-up.sh`).
접속할 수 없으면 각 테스트 클래스가 skip된다.

테스트 데이터는 실제 테이블에 들어가므로, 식별 가능한 값만 쓰고 tearDown에서 지운다.
  - raw_articles: url이 'https://test.invalid/'로 시작
  - digests: keyword가 '__test__'  (send_log는 FK CASCADE로 함께 삭제)
"""
import unittest
from datetime import date

import psycopg

from config import config
from db.db import get_conn, init_db

TEST_KEYWORD = "__test__"
TEST_URL_PREFIX = "https://test.invalid/"


def db_available() -> bool:
    """DB에 접속 가능한지 확인. 테스트 skip 조건으로 쓴다."""
    try:
        with psycopg.connect(config.DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


requires_db = unittest.skipUnless(
    db_available(),
    f"DB에 접속할 수 없다 - scripts/db-up.ps1 로 먼저 띄울 것 ({config.DATABASE_URL})",
)


def cleanup_test_rows() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM raw_articles WHERE url LIKE %s", (TEST_URL_PREFIX + "%",))
        # send_log는 digests FK가 ON DELETE CASCADE라 함께 지워진다.
        conn.execute("DELETE FROM digests WHERE keyword = %s", (TEST_KEYWORD,))


class DbTestCase(unittest.TestCase):
    """스키마 준비 + 테스트 데이터 정리를 담당하는 베이스 클래스."""

    @classmethod
    def setUpClass(cls):
        init_db()  # 스키마가 없으면 만든다 (있으면 no-op)

    def setUp(self):
        self.today = date.today().isoformat()

    def tearDown(self):
        cleanup_test_rows()

    def article(self, n: int, **overrides) -> dict:
        """테스트용 기사 1건. url은 tearDown이 찾을 수 있는 접두사를 쓴다."""
        return {
            "title": f"테스트 기사 {n}",
            "description": "설명",
            "url": f"{TEST_URL_PREFIX}{n}",
            "source": "test",
            "published_at": "2026-08-14T00:00:00Z",
        } | overrides
