"""SQLite 접근 헬퍼 (구현 항목 #2). 파이프라인의 모든 노드가 이 모듈을 통해서만 DB에 접근한다."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import config

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db() -> None:
    """DB 파일과 상위 디렉터리, 테이블을 생성한다. 앱 시작 시 1회 호출."""
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_articles(digest_date: str, articles: list[dict]) -> int:
    """중복 URL은 무시(UNIQUE 제약)하고 신규 기사만 저장. 저장된 신규 건수를 반환."""
    inserted = 0
    with get_conn() as conn:
        for a in articles:
            try:
                conn.execute(
                    """INSERT INTO raw_articles
                       (digest_date, title, description, url, source, published_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        digest_date,
                        a.get("title", ""),
                        a.get("description", ""),
                        a.get("url", ""),
                        a.get("source", ""),
                        a.get("published_at", ""),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # 이미 저장된 기사(URL 중복) -> 스킵
                continue
    return inserted


def save_digest(digest_date: str, keyword: str, summary: list[dict], insight: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO digests (digest_date, keyword, summary_json, insight_json)
               VALUES (?, ?, ?, ?)""",
            (digest_date, keyword, json.dumps(summary, ensure_ascii=False), json.dumps(insight, ensure_ascii=False)),
        )
        return cur.lastrowid


def log_send(digest_id: int, channel: str, recipient: str, status: str, error: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO send_log (digest_id, channel, recipient, status, error)
               VALUES (?, ?, ?, ?, ?)""",
            (digest_id, channel, recipient, status, error),
        )
