"""DB 접속 테스트.

두 부분으로 나뉜다.
  - ConnectionStringTest: DATABASE_URL 조립 규칙. DB 없이도 돌아간다.
  - ConnectTest: 실제 접속/인코딩/실패 케이스. DB가 없으면 skip.

    python -m unittest tests.test_db_connection
"""
import importlib
import os
import unittest
from unittest import mock
from urllib.parse import unquote, urlparse

import psycopg

from config import config
from db.db import get_conn, init_db
from tests.dbhelpers import requires_db

# 조립 규칙 테스트에서 덮어써야 하는 변수들
DB_ENV_KEYS = (
    "DATABASE_URL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


class ConnectionStringTest(unittest.TestCase):
    """config.py가 접속 문자열을 만드는 규칙 (실제 접속은 하지 않는다)."""

    def tearDown(self):
        # 다른 테스트가 원래 설정을 보도록 모듈을 되돌린다.
        importlib.reload(importlib.import_module("config"))

    def _database_url(self, **env) -> str:
        """주어진 환경변수만 적용된 상태에서 config를 다시 읽어 DATABASE_URL을 얻는다."""
        with mock.patch.dict(os.environ, env):
            for key in DB_ENV_KEYS:
                if key not in env:
                    os.environ.pop(key, None)  # patch.dict가 종료 시 복구한다
            module = importlib.reload(importlib.import_module("config"))
            return module.config.DATABASE_URL

    def test_POSTGRES_변수들로_조립한다(self):
        url = self._database_url(
            POSTGRES_HOST="db.example.com",
            POSTGRES_PORT="6543",
            POSTGRES_DB="mydb",
            POSTGRES_USER="myuser",
            POSTGRES_PASSWORD="mypass",
        )

        self.assertEqual("postgresql://myuser:mypass@db.example.com:6543/mydb", url)

    def test_아무것도_없으면_로컬_기본값을_쓴다(self):
        self.assertEqual("postgresql://briefyi:briefyi@localhost:5432/briefyi", self._database_url())

    def test_DATABASE_URL이_있으면_그_값을_그대로_쓴다(self):
        given = "postgresql://other:pw@managed.example.com:5432/prod?sslmode=require"

        url = self._database_url(DATABASE_URL=given, POSTGRES_HOST="ignored", POSTGRES_DB="ignored")

        self.assertEqual(given, url)

    def test_비밀번호의_특수문자를_URL_인코딩한다(self):
        url = self._database_url(POSTGRES_USER="us er", POSTGRES_PASSWORD="p@ss:w/rd")

        self.assertEqual("postgresql://us+er:p%40ss%3Aw%2Frd@localhost:5432/briefyi", url)
        # 인코딩되지 않으면 '@'와 '/' 때문에 호스트/DB 이름 파싱이 깨진다.
        parsed = urlparse(url)
        self.assertEqual("localhost", parsed.hostname)
        self.assertEqual("briefyi", parsed.path.lstrip("/"))
        self.assertEqual("p@ss:w/rd", unquote(parsed.password))


@requires_db
class ConnectTest(unittest.TestCase):
    """실제 컨테이너에 붙어서 확인."""

    def test_접속해서_쿼리할_수_있다(self):
        with get_conn() as conn:
            row = conn.execute("SELECT 1 AS one").fetchone()

        self.assertEqual({"one": 1}, row, "row_factory=dict_row 이므로 dict로 돌아온다")

    def test_접속한_DB와_계정이_설정과_일치한다(self):
        parsed = urlparse(config.DATABASE_URL)

        with get_conn() as conn:
            row = conn.execute(
                "SELECT current_database() AS db, current_user AS usr, version() AS ver"
            ).fetchone()

        self.assertEqual(parsed.path.lstrip("/"), row["db"])
        self.assertEqual(parsed.username, row["usr"])
        self.assertIn("PostgreSQL 16", row["ver"])

    def test_서버_인코딩이_UTF8이고_한글이_왕복한다(self):
        with get_conn() as conn:
            encoding = conn.execute("SHOW server_encoding").fetchone()["server_encoding"]
            row = conn.execute("SELECT %s::text AS ko", ("한글 인코딩 확인",)).fetchone()

        self.assertEqual("UTF8", encoding)
        self.assertEqual("한글 인코딩 확인", row["ko"])

    def test_init_db는_여러_번_호출해도_안전하다(self):
        init_db()
        init_db()  # CREATE TABLE IF NOT EXISTS 라 예외가 없어야 한다

        with get_conn() as conn:
            rows = conn.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = 'public'"""
            ).fetchall()

        self.assertEqual(
            {
                "raw_articles",
                "digests",
                "send_log",
                # db/vector_schema.sql (RAG)
                "article_chunks",
                "chunk_embeddings",
                "article_topics",
                "article_events",
                "article_event_arguments",
                "article_event_index_status",
            },
            {r["table_name"] for r in rows},
        )

    def test_없는_DB_이름으로는_접속에_실패한다(self):
        parsed = urlparse(config.DATABASE_URL)
        bad = config.DATABASE_URL.replace(parsed.path, "/does_not_exist_db")

        with self.assertRaises(psycopg.OperationalError):
            psycopg.connect(bad, connect_timeout=3)

    def test_열려_있지_않은_포트로는_접속에_실패한다(self):
        with self.assertRaises(psycopg.OperationalError):
            psycopg.connect("postgresql://briefyi:briefyi@localhost:59999/briefyi", connect_timeout=3)


if __name__ == "__main__":
    unittest.main()
