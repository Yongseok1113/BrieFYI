"""트리거 CLI.

사용 예:
    python -m trigger                            # hello를 10초마다 (Ctrl+C로 종료)
    python -m trigger --interval 5               # 5초마다
    python -m trigger --duration 25              # 25초만 돌고 종료
    python -m trigger --once                     # 1회만 실행
"""
import argparse
import logging
import threading

from trigger.jobs import JOBS
from trigger.scheduler import Scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="주기 실행 트리거")
    parser.add_argument("--job", default="hello", choices=sorted(JOBS), help="실행할 작업 이름")
    parser.add_argument("--interval", type=float, default=10.0, help="실행 주기(초), 기본 10")
    parser.add_argument("--duration", type=float, help="이 시간(초) 뒤 자동 종료. 없으면 무한 실행")
    parser.add_argument("--once", action="store_true", help="주기 실행 없이 1회만 실행")
    args = parser.parse_args(argv)

    func = JOBS[args.job]

    if args.once:
        func()
        return 0

    scheduler = Scheduler()
    job = scheduler.add_job(func, args.interval, name=args.job)

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

    logger.info("%s 실행 %d회, 실패 %d회", job.name, job.run_count, job.error_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
