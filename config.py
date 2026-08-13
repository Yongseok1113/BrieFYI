"""전역 설정 로더. .env 파일 값을 읽어 파이프라인 전체에서 공유한다."""
import os
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

    # 저장소
    DB_PATH = os.getenv("DB_PATH", "./data/pipeline.db")


config = Config()
