"""트리거 스케줄러 테스트.

기본 테스트는 가짜 시계를 주입해 실제 대기 없이 '10초 주기'를 검증한다(즉시 완료).
실제 시간으로 10초 주기를 확인하는 통합 테스트는 약 21초가 걸리므로
RUN_SLOW_TESTS=1 환경변수가 있을 때만 실행된다.

    python -m unittest tests.test_scheduler            # 빠른 테스트
    RUN_SLOW_TESTS=1 python -m unittest tests.test_scheduler   # 실제 10초 주기까지
"""
import os
import threading
import time
import unittest

from trigger.jobs import hello_world
from trigger.scheduler import Scheduler


class FakeClock:
    """가상 시계. sleeper가 호출될 때마다 시간을 건너뛰고, 한계 시각에서 중지 신호를 준다."""

    def __init__(self, stop_at: float):
        self.now = 0.0
        self.stop_at = stop_at

    def monotonic(self) -> float:
        return self.now

    def sleeper(self, timeout: float) -> bool:
        # 최소 진행량을 두어 timeout=0에서 무한 루프가 되지 않게 한다.
        self.now += max(timeout, 0.001)
        return self.now >= self.stop_at

    def advance(self, seconds: float) -> None:
        self.now += seconds


class HelloWorldJobTest(unittest.TestCase):
    def test_hello_world_반환값(self):
        self.assertIn("hello world", hello_world())


class IntervalTriggerTest(unittest.TestCase):
    """가짜 시계 기반 주기 검증."""

    def _record_runs(self, clock: FakeClock, **job_kwargs) -> list[float]:
        """작업이 실행된 (가상) 시각 목록을 돌려준다."""
        fired: list[float] = []
        scheduler = Scheduler(monotonic=clock.monotonic, sleeper=clock.sleeper)
        scheduler.add_job(lambda: fired.append(clock.now), interval=10.0, name="hello", **job_kwargs)
        scheduler.run_loop()
        return fired

    def test_10초마다_실행된다(self):
        clock = FakeClock(stop_at=35.0)

        fired = self._record_runs(clock)

        self.assertEqual([0.0, 10.0, 20.0, 30.0], fired)
        gaps = [b - a for a, b in zip(fired, fired[1:])]
        self.assertTrue(all(gap == 10.0 for gap in gaps), gaps)

    def test_run_immediately_False면_10초_뒤_첫_실행(self):
        clock = FakeClock(stop_at=35.0)

        fired = self._record_runs(clock, run_immediately=False)

        self.assertEqual([10.0, 20.0, 30.0], fired)

    def test_작업이_오래_걸려도_주기가_밀리지_않는다(self):
        clock = FakeClock(stop_at=35.0)
        fired: list[float] = []

        def slow_job():
            fired.append(clock.now)
            clock.advance(5.0)  # 5초 걸리는 작업

        scheduler = Scheduler(monotonic=clock.monotonic, sleeper=clock.sleeper)
        scheduler.add_job(slow_job, interval=10.0, name="slow")
        scheduler.run_loop()

        self.assertEqual([0.0, 10.0, 20.0, 30.0], fired)

    def test_한_주기보다_오래_걸리면_밀린_실행은_건너뛴다(self):
        clock = FakeClock(stop_at=70.0)
        fired: list[float] = []

        def very_slow_job():
            fired.append(clock.now)
            clock.advance(25.0)  # 주기(10초)를 두 번 넘기는 작업

        scheduler = Scheduler(monotonic=clock.monotonic, sleeper=clock.sleeper)
        scheduler.add_job(very_slow_job, interval=10.0, name="very_slow")
        scheduler.run_loop()

        # 몰아서 따라잡지 않고 다음 슬롯(30, 60초)으로 정렬된다.
        self.assertEqual([0.0, 30.0, 60.0], fired)

    def test_작업이_예외를_던져도_다음_주기에_계속_실행된다(self):
        clock = FakeClock(stop_at=35.0)
        calls = []

        def failing_job():
            calls.append(clock.now)
            raise RuntimeError("boom")

        scheduler = Scheduler(monotonic=clock.monotonic, sleeper=clock.sleeper)
        job = scheduler.add_job(failing_job, interval=10.0, name="failing")
        with self.assertLogs("trigger.scheduler", level="ERROR"):
            scheduler.run_loop()

        self.assertEqual([0.0, 10.0, 20.0, 30.0], calls)
        self.assertEqual(0, job.run_count)
        self.assertEqual(4, job.error_count)
        self.assertEqual("boom", job.last_error)

    def test_여러_작업이_각자의_주기로_실행된다(self):
        clock = FakeClock(stop_at=31.0)
        fired: list[tuple[str, float]] = []

        scheduler = Scheduler(monotonic=clock.monotonic, sleeper=clock.sleeper)
        scheduler.add_job(lambda: fired.append(("a", clock.now)), interval=10.0, name="a")
        scheduler.add_job(lambda: fired.append(("b", clock.now)), interval=30.0, name="b")
        scheduler.run_loop()

        self.assertEqual([("a", t) for t in (0.0, 10.0, 20.0, 30.0)], [f for f in fired if f[0] == "a"])
        self.assertEqual([("b", 0.0), ("b", 30.0)], [f for f in fired if f[0] == "b"])

    def test_interval이_0이하면_거부한다(self):
        scheduler = Scheduler()
        with self.assertRaises(ValueError):
            scheduler.add_job(hello_world, interval=0)


class RealTimeTriggerTest(unittest.TestCase):
    """실제 스레드/시계로 도는지 확인 (짧은 주기로 빠르게)."""

    def test_백그라운드_스레드에서_주기_실행되고_stop으로_멈춘다(self):
        third_run = threading.Event()
        fired: list[float] = []

        def job():
            fired.append(time.monotonic())
            if len(fired) >= 3:
                third_run.set()

        scheduler = Scheduler()
        tracked = scheduler.add_job(job, interval=0.05, name="fast")
        scheduler.start()
        try:
            self.assertTrue(third_run.wait(5.0), "3회 실행되지 않았다")
        finally:
            scheduler.stop()

        self.assertGreaterEqual(tracked.run_count, 3)
        count_at_stop = tracked.run_count
        time.sleep(0.2)
        self.assertEqual(count_at_stop, tracked.run_count, "stop 후에도 실행됐다")


@unittest.skipUnless(os.getenv("RUN_SLOW_TESTS"), "RUN_SLOW_TESTS=1일 때만 실행 (약 21초)")
class TenSecondIntervalTest(unittest.TestCase):
    """실제 시간으로 hello_world가 10초마다 실행되는지 검증."""

    def test_hello_world가_실제로_10초마다_실행된다(self):
        fired: list[float] = []
        third_run = threading.Event()

        def job():
            hello_world()
            fired.append(time.monotonic())
            if len(fired) >= 3:
                third_run.set()

        scheduler = Scheduler()
        scheduler.add_job(job, interval=10.0, name="hello")
        scheduler.start()
        try:
            self.assertTrue(third_run.wait(25.0), "25초 안에 3회 실행되지 않았다")
        finally:
            scheduler.stop()

        gaps = [b - a for a, b in zip(fired, fired[1:])]
        self.assertEqual(2, len(gaps))
        for gap in gaps:
            self.assertAlmostEqual(10.0, gap, delta=0.5, msg=f"실행 간격 {gap:.3f}초")


if __name__ == "__main__":
    unittest.main()
