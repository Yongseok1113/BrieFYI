"""
구독자(subscriber) 저장 계층 - 팀 공용 Postgres 사용.
"""

from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from db.db import get_conn


def upsert_subscriber(email: str, categories: list[str]) -> dict:
    """
    이메일로 구독자를 저장한다.
    이미 있는 이메일이면 관심 분야만 최신 값으로 갱신한다(재신청 허용).

    Returns:
        {"created": True/False} - 신규 등록이었는지 여부
    """

    now = datetime.now(timezone.utc)

    with get_conn() as conn:

        existing = conn.execute(
            "SELECT id FROM subscribers WHERE email = %s", (email,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE subscribers SET categories = %s, updated_at = %s WHERE email = %s",
                (Jsonb(categories), now, email),
            )
            return {"created": False}

        conn.execute(
            """
            INSERT INTO subscribers (email, categories, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            """,
            (email, Jsonb(categories), now, now),
        )
        return {"created": True}


def delete_subscriber(email: str) -> bool:
    """
    이메일로 구독자를 삭제한다.

    Returns:
        True  - 실제로 있던 구독자를 지웠음
        False - 애초에 그 이메일로 등록된 구독자가 없었음
    """

    with get_conn() as conn:
        cur = conn.execute("DELETE FROM subscribers WHERE email = %s", (email,))
        return cur.rowcount > 0


def get_subscriber_count() -> int:
    """
    현재 구독자 수를 반환한다 (헬스체크/확인용).
    """

    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM subscribers").fetchone()
        return row["cnt"]