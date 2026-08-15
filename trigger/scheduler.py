"""간격 기반 트리거 스케줄러 (설계문서 3.1 트리거 계층).

외부 의존성 없이 표준 라이브러리만 사용한다. 등록된 작업을 각자의 주기마다
호출하고, 작업이 예외를 던져도 스케줄러는 죽지 않고 다음 주기를 계속 돈다.

- 고정 주기(fixed-rate): 작업 실행에 걸린 시간과 무관하게 다음 실행은
  '이전 예정 시각 + interval'로 잡는다. 즉 10초 주기면 작업이 1초 걸려도
  10초마다 실행된다. 한 주기를 통째로 넘길 만큼 늦어지면 밀린 실행은 건너뛴다.
- 단일 스레드: 모든 작업을 하나의 루프 스레드에서 순차 실행한다. 긴 작업은
  다른 작업을 지연시키므로, 오래 걸리는 파이프라인은 주기를 넉넉히 잡는다.
- monotonic/sleeper를 주입할 수 있어(테스트용 가짜 시계) 실시간 대기 없이
  주기 동작을 검증할 수 있다.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """등록된 하나의 주기 작업과 그 실행 상태."""

    name: str
    func: Callable[[], object]
    interval: float
    next_run: float = 0.0
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_result: object = field(default=None, repr=False)


class Scheduler:
    """작업을 주기적으로 실행하는 스케줄러."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Optional[Callable[[float], bool]] = None,
    ):
        """monotonic은 현재 시각(초), sleeper(timeout)는 대기 함수다.

        sleeper는 대기 중 중지 요청을 받으면 True를 반환해야 한다. 기본값은
        내부 stop 이벤트의 wait이며, 테스트에서는 가짜 시계를 넘겨 즉시 진행시킨다.
        """
        self._monotonic = monotonic
        self._stop_event = threading.Event()
        self._sleeper = sleeper or self._stop_event.wait
        self._jobs: list[Job] = []
        self._thread: Optional[threading.Thread] = None

    @property
    def jobs(self) -> list[Job]:
        return list(self._jobs)

    def add_job(
        self,
        func: Callable[[], object],
        interval: float,
        name: Optional[str] = None,
        run_immediately: bool = True,
    ) -> Job:
        """작업을 등록한다.

        run_immediately=True면 시작 직후 1회 실행하고 그 뒤부터 interval 주기로,
        False면 interval 만큼 기다린 뒤 첫 실행을 한다.
        """
        if interval <= 0:
            raise ValueError("interval은 0보다 커야 한다")

        now = self._monotonic()
        job = Job(
            name=name or getattr(func, "__name__", "job"),
            func=func,
            interval=float(interval),
            next_run=now if run_immediately else now + interval,
        )
        self._jobs.append(job)
        logger.info("작업 등록: %s (%.3g초 주기)", job.name, job.interval)
        return job

    def start(self) -> None:
        """백그라운드 스레드에서 스케줄 루프를 돌린다."""
        if self._thread is not None:
            raise RuntimeError("이미 실행 중이다")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_loop, name="trigger", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """중지 요청 후 루프 스레드가 끝날 때까지 기다린다."""
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)

    def run_loop(self) -> None:
        """중지 요청이 올 때까지 예정된 작업을 실행하는 메인 루프."""
        logger.info("트리거 시작 (작업 %d개)", len(self._jobs))
        while not self._stop_event.is_set():
            now = self._monotonic()
            for job in self._jobs:
                if job.next_run <= now:
                    self._run_job(job)

            timeout = self._next_timeout()
            if self._sleeper(timeout):
                break
        logger.info("트리거 종료")

    def run_forever(self) -> None:
        """포그라운드 실행. Ctrl+C를 받으면 정상 종료한다."""
        try:
            self.run_loop()
        except KeyboardInterrupt:
            logger.info("중단 요청(Ctrl+C) 수신")
            self._stop_event.set()

    def _run_job(self, job: Job) -> None:
        scheduled_at = job.next_run
        try:
            job.last_result = job.func()
        except Exception as exc:  # noqa: BLE001 - 한 작업의 실패로 트리거가 멈추면 안 된다
            job.error_count += 1
            job.last_error = str(exc)
            logger.exception("작업 실패: %s", job.name)
        else:
            job.run_count += 1

        # 고정 주기 유지. 한 주기 이상 밀렸으면 밀린 실행은 버리고 다음 슬롯으로.
        job.next_run = scheduled_at + job.interval
        now = self._monotonic()
        if job.next_run <= now:
            skipped = int((now - job.next_run) // job.interval) + 1
            job.next_run += skipped * job.interval
            logger.warning("%s: 실행이 지연되어 %d회 건너뜀", job.name, skipped)

    def _next_timeout(self) -> float:
        if not self._jobs:
            return 1.0
        return max(0.0, min(job.next_run for job in self._jobs) - self._monotonic())
