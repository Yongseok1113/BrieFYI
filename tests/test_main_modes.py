"""main.py 실행 모드 테스트 (single / trigger).

run_pipeline과 init_db를 대체(mock)하므로 외부 API 호출·DB 접근·이메일 발송이 없다.
trigger 모드의 주기 검증은 가짜 시계를 주입해 실제 대기 없이 끝난다.

    python -m unittest tests.test_main_modes
"""
import unittest
from unittest import mock

import main
from tests.test_scheduler import FakeClock
from trigger.scheduler import Scheduler


def success_result() -> dict:
    """run_pipeline이 정상 종료했을 때의 최소 상태."""
    return {
        "raw_articles": [{"title": "t"}],
        "inserted_count": 1,
        "insight": {"insights": ["i1", "i2"]},
        "send_result": {"id": "mock"},
        "error": None,
    }


class SingleModeTest(unittest.TestCase):
    def test_1회_실행하고_0을_반환한다(self):
        with mock.patch.object(main, "init_db") as init_db, mock.patch.object(
            main, "run_pipeline", return_value=success_result()
        ) as run_pipeline:
            exit_code = main.main(["--mode", "single", "--keyword", "AI", "--days", "2", "--max-results", "5"])

        self.assertEqual(0, exit_code)
        init_db.assert_called_once()
        run_pipeline.assert_called_once_with("AI", 2, 5)

    def test_mode를_생략하면_config_기본값이_쓰인다(self):
        args = main.build_parser().parse_args([])
        self.assertEqual(main.config.RUN_MODE, args.mode)
        self.assertEqual(main.config.TRIGGER_INTERVAL_SECONDS, args.interval)

    def test_파이프라인_에러면_1을_반환한다(self):
        failed = success_result() | {"error": "발송 실패"}
        with mock.patch.object(main, "init_db"), mock.patch.object(
            main, "run_pipeline", return_value=failed
        ) as run_pipeline:
            exit_code = main.main(["--mode", "single"])

        self.assertEqual(1, exit_code)
        run_pipeline.assert_called_once()


class TriggerModeTest(unittest.TestCase):
    """가짜 시계로 '설정된 주기마다 main 로직이 실행되는지' 검증."""

    def _fake_clock_scheduler(self, clock: FakeClock):
        """main이 생성하는 Scheduler를 가짜 시계 버전으로 바꿔주는 patcher."""
        return mock.patch.object(
            main, "Scheduler", lambda: Scheduler(monotonic=clock.monotonic, sleeper=clock.sleeper)
        )

    def test_설정된_주기마다_파이프라인이_반복_실행된다(self):
        clock = FakeClock(stop_at=35.0)
        fired: list[float] = []

        def fake_pipeline(*_args):
            fired.append(clock.now)
            return success_result()

        with mock.patch.object(main, "init_db"), mock.patch.object(
            main, "run_pipeline", side_effect=fake_pipeline
        ), self._fake_clock_scheduler(clock):
            exit_code = main.main(["--mode", "trigger", "--interval", "10"])

        self.assertEqual(0, exit_code)
        self.assertEqual([0.0, 10.0, 20.0, 30.0], fired)

    def test_주기_실행에_CLI_파라미터가_그대로_전달된다(self):
        clock = FakeClock(stop_at=15.0)
        with mock.patch.object(main, "init_db"), mock.patch.object(
            main, "run_pipeline", return_value=success_result()
        ) as run_pipeline, self._fake_clock_scheduler(clock):
            main.main(["--mode", "trigger", "--interval", "10", "--keyword", "반도체", "--days", "3", "--max-results", "7"])

        self.assertEqual([mock.call("반도체", 3, 7)] * 2, run_pipeline.call_args_list)

    def test_한_주기가_실패해도_다음_주기는_계속_실행된다(self):
        clock = FakeClock(stop_at=35.0)
        with mock.patch.object(main, "init_db"), mock.patch.object(
            main, "run_pipeline", side_effect=RuntimeError("GNEWS_API_KEY 없음")
        ) as run_pipeline, self._fake_clock_scheduler(clock):
            with self.assertLogs("trigger.scheduler", level="ERROR"):
                exit_code = main.main(["--mode", "trigger", "--interval", "10"])

        self.assertEqual(0, exit_code, "실패해도 트리거는 정상 종료해야 한다")
        self.assertEqual(4, run_pipeline.call_count)

    def test_duration이_지나면_스레드를_정리하고_종료한다(self):
        """실제 시계/스레드 경로(--duration)도 도는지 짧은 주기로 확인."""
        with mock.patch.object(main, "init_db"), mock.patch.object(
            main, "run_pipeline", return_value=success_result()
        ) as run_pipeline:
            exit_code = main.main(["--mode", "trigger", "--interval", "0.05", "--duration", "0.3"])

        self.assertEqual(0, exit_code)
        self.assertGreaterEqual(run_pipeline.call_count, 3)
        count_at_exit = run_pipeline.call_count
        import time

        time.sleep(0.2)
        self.assertEqual(count_at_exit, run_pipeline.call_count, "종료 후에도 실행됐다")


class DigestJobTest(unittest.TestCase):
    def test_trigger_jobs의_digest가_main_run_digest를_호출한다(self):
        from trigger.jobs import JOBS

        self.assertIn("digest", JOBS)
        with mock.patch.object(main, "init_db"), mock.patch.object(
            main, "run_pipeline", return_value=success_result()
        ) as run_pipeline, mock.patch("db.db.init_db"):
            JOBS["digest"]()

        run_pipeline.assert_called_once_with(
            main.config.NEWS_KEYWORD, main.config.NEWS_LOOKBACK_DAYS, main.config.NEWS_MAX_RESULTS
        )


if __name__ == "__main__":
    unittest.main()
