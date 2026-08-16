"""CLI 진입점 (구현 항목 #7~#8).

두 가지 실행 모드를 지원한다.

- single (기본): 파이프라인을 1회 실행하고 종료한다. cron/GitHub Actions처럼
  외부 스케줄러가 이 스크립트를 정기 실행하는 방식.
- trigger: 프로세스를 띄워둔 채 `--interval` 주기마다 파이프라인을 반복 실행한다
  (설계문서 3.1 트리거 계층). 한 번의 실행이 실패해도 다음 주기는 계속 돈다.
"""
import argparse
import logging
import threading

from config import config
from db.db import init_db
from graph.pipeline import run_pipeline
from trigger.scheduler import Scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_digest(keyword: str, days: int, max_results: int) -> dict:
    """파이프라인 1회 실행. 실패하면 RuntimeError를 던진다.

    트리거 모드에서는 이 예외를 스케줄러가 잡아 로그로 남기고 다음 주기를 계속 돈다.
    """
    logger.info("파이프라인 시작: keyword=%s, days=%s", keyword, days)
    result = run_pipeline(keyword, days, max_results)

    if result.get("error"):
        raise RuntimeError(result["error"])

    logger.info(
        "완료: 수집 %d건, 신규저장 %d건, 인사이트 %d개, 발송 결과 %s",
        len(result["raw_articles"]),
        result["inserted_count"],
        len(result["insight"].get("insights", [])),
        result["send_result"],
    )
    return result


def run_single(args: argparse.Namespace) -> int:
    try:
        run_digest(args.keyword, args.days, args.max_results)
    except Exception as exc:  # noqa: BLE001 - 실패 원인을 로그로 남기고 종료 코드로 알린다
        logger.error("파이프라인 실패: %s", exc)
        return 1
    return 0


def run_trigger(args: argparse.Namespace) -> int:
    """`--interval` 주기마다 파이프라인을 반복 실행한다."""
    scheduler = Scheduler()
    job = scheduler.add_job(
        lambda: run_digest(args.keyword, args.days, args.max_results),
        interval=args.interval,
        name="digest",
    )
    logger.info("트리거 모드: %.3g초 주기로 파이프라인을 반복 실행한다 (Ctrl+C 종료)", args.interval)

    if args.duration is None:
        scheduler.run_forever()
    else:
        scheduler.start()
        try:
            threading.Event().wait(args.duration)
        except KeyboardInterrupt:
            logger.info("중단 요청(Ctrl+C) 수신")
        finally:
            scheduler.stop()

    logger.info("트리거 종료: 성공 %d회, 실패 %d회", job.run_count, job.error_count)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="뉴스·기술문서 다이제스트 파이프라인 실행")
    parser.add_argument(
        "--mode",
        choices=("single", "trigger"),
        default=config.RUN_MODE,
        help="single=1회 실행 후 종료, trigger=주기 반복 실행 (기본: %(default)s)",
    )
    parser.add_argument("--keyword", default=config.NEWS_KEYWORD)
    parser.add_argument("--days", type=int, default=config.NEWS_LOOKBACK_DAYS)
    parser.add_argument("--max-results", type=int, default=config.NEWS_MAX_RESULTS)
    parser.add_argument(
        "--interval",
        type=float,
        default=config.TRIGGER_INTERVAL_SECONDS,
        help="trigger 모드 실행 주기(초) (기본: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="trigger 모드에서 이 시간(초) 뒤 자동 종료. 없으면 무한 실행",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    init_db()

    if args.mode == "trigger":
        return run_trigger(args)
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
