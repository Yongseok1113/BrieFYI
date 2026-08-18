"""PostgreSQL 접근 헬퍼 (구현 항목 #2). 파이프라인의 모든 노드가 이 모듈을 통해서만 DB에 접근한다.

접속 정보는 config.DATABASE_URL 하나로 정해진다. DB는 docker-compose의 `db` 서비스로 띄운다
(scripts/db-up.ps1 또는 scripts/db-up.sh).
"""
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from config import config

SCHEMA_PATHS = (
    Path(__file__).parent / "schema.sql",
    Path(__file__).parent / "vector_schema.sql",
)


def init_db() -> None:
    """테이블을 생성한다. 앱 시작 시 1회 호출.

    컨테이너 최초 기동 시 Dockerfile.db의 초기화 스크립트가 이미 같은 스키마를 적용하지만,
    스키마가 IF NOT EXISTS라 여기서 다시 실행해도 안전하다(기존 볼륨/외부 DB 대응).
    파라미터가 없는 쿼리는 여러 문장을 한 번에 실행할 수 있다.
    """
    with get_conn() as conn:
        for schema_path in SCHEMA_PATHS:
            conn.execute(schema_path.read_text(encoding="utf-8"))


@contextmanager
def get_conn():
    """커넥션 컨텍스트. 정상 종료 시 커밋, 예외 시 롤백."""
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_articles(digest_date: str, articles: list[dict]) -> int:
    """중복 URL은 무시(UNIQUE 제약)하고 신규 기사만 저장. 저장된 신규 건수를 반환."""
    inserted = 0
    with get_conn() as conn, conn.cursor() as cur:
        for a in articles:
            cur.execute(
                """INSERT INTO raw_articles
                   (digest_date, title, description, url, source, published_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (url) DO NOTHING""",
                (
                    digest_date,
                    a.get("title", ""),
                    a.get("description", ""),
                    a.get("url", ""),
                    a.get("source", ""),
                    # 빈 문자열은 TIMESTAMPTZ 캐스팅에 실패하므로 NULL로 넣는다.
                    a.get("published_at") or None,
                ),
            )
            inserted += cur.rowcount  # 중복이면 0
    return inserted


def save_digest(digest_date: str, keyword: str, summary: list[dict], insight: dict) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO digests (digest_date, keyword, summary_json, insight_json)
               VALUES (%s, %s, %s, %s)
               RETURNING id""",
            (digest_date, keyword, Jsonb(summary), Jsonb(insight)),
        )
        return cur.fetchone()["id"]


def log_send(digest_id: int, channel: str, recipient: str, status: str, error: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO send_log (digest_id, channel, recipient, status, error)
               VALUES (%s, %s, %s, %s, %s)""",
            (digest_id, channel, recipient, status, error),
        )
