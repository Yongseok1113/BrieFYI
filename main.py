"""CLI 진입점 (구현 항목 #7~#8). 스케줄러(cron 등)는 이 스크립트를 정기 실행하기만 하면 된다."""
import argparse
import logging

from config import config
from db.db import init_db
from graph.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="뉴스·기술문서 다이제스트 파이프라인 실행")
    parser.add_argument("--keyword", default=config.NEWS_KEYWORD)
    parser.add_argument("--days", type=int, default=config.NEWS_LOOKBACK_DAYS)
    parser.add_argument("--max-results", type=int, default=config.NEWS_MAX_RESULTS)
    args = parser.parse_args()

    init_db()
    logger.info("파이프라인 시작: keyword=%s, days=%s", args.keyword, args.days)

    result = run_pipeline(args.keyword, args.days, args.max_results)

    if result.get("error"):
        logger.error("파이프라인 실패: %s", result["error"])
        raise SystemExit(1)

    logger.info(
        "완료: 수집 %d건, 신규저장 %d건, 인사이트 %d개, 발송 결과 %s",
        len(result["raw_articles"]),
        result["inserted_count"],
        len(result["insight"].get("insights", [])),
        result["send_result"],
    )


if __name__ == "__main__":
    main()
