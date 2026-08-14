"""트리거가 실행할 작업 모음.

hello_world는 트리거 동작 확인용이고, digest는 main.run_digest(다이제스트 파이프라인)를
호출한다. 새 작업은 함수를 추가하고 JOBS에 등록하면 CLI(`python -m trigger --job <이름>`)에서
바로 쓸 수 있다.
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def hello_world() -> str:
    """가장 단순한 작업. 실행 시각과 함께 인사말을 로그로 남기고 반환한다."""
    message = f"hello world ({datetime.now().strftime('%H:%M:%S')})"
    logger.info(message)
    return message


def run_digest_job() -> dict:
    """다이제스트 파이프라인 1회 실행 (main.run_digest 호출).

    main은 langgraph 등 무거운 의존성을 끌어오므로, hello만 쓸 때 부담이 없도록
    함수 안에서 지연 임포트한다. (main -> trigger.scheduler 순환 임포트도 함께 피한다)
    """
    from config import config
    from db.db import init_db
    from main import run_digest

    init_db()
    return run_digest(config.NEWS_KEYWORD, config.NEWS_LOOKBACK_DAYS, config.NEWS_MAX_RESULTS)


JOBS = {
    "hello": hello_world,
    "digest": run_digest_job,
}
