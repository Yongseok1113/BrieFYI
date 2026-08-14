"""전역 설정 로더. .env 파일 값을 읽어 파이프라인 전체에서 공유한다."""
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


class Config:
    # GNews
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
    GNEWS_BASE_URL = "https://gnews.io/api/v4/search"

    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    # Email (Resend)
    RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", "")
    EMAIL_TO = os.getenv("EMAIL_TO", "")

    # 파이프라인 파라미터
    NEWS_KEYWORD = os.getenv("NEWS_KEYWORD", "AI")
    NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "1"))
    NEWS_MAX_RESULTS = int(os.getenv("NEWS_MAX_RESULTS", "10"))

    # 실행 모드 / 트리거
    # RUN_MODE: single=1회 실행 후 종료(cron 등 외부 스케줄러용), trigger=주기 반복 실행
    RUN_MODE = os.getenv("RUN_MODE", "single")
    TRIGGER_INTERVAL_SECONDS = float(os.getenv("TRIGGER_INTERVAL_SECONDS", "86400"))

    # 저장소 (PostgreSQL, docker-compose의 db 서비스)
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "briefyi")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "briefyi")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "briefyi")
    # DATABASE_URL을 직접 주면 그 값을 그대로 쓴다(컨테이너 안에서는 host가 db).
    DATABASE_URL = os.getenv("DATABASE_URL") or (
        f"postgresql://{quote_plus(POSTGRES_USER)}:{quote_plus(POSTGRES_PASSWORD)}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


config = Config()
