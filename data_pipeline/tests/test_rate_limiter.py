from data_pipeline.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_allows_up_to_max_requests_without_sleep():
    clock = FakeClock()
    limiter = RateLimiter(3, 60, sleep_fn=clock.sleep, time_fn=clock.time)

    for _ in range(3):
        limiter.acquire()

    assert clock.sleeps == []
    assert limiter.remaining() == 0


def test_blocks_when_window_full():
    clock = FakeClock()
    limiter = RateLimiter(2, 60, sleep_fn=clock.sleep, time_fn=clock.time)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # 3번째는 대기해야 함

    assert clock.sleeps  # 최소 한 번은 sleep 호출됨


def test_old_requests_expire_out_of_window():
    clock = FakeClock()
    limiter = RateLimiter(1, 10, sleep_fn=clock.sleep, time_fn=clock.time)

    limiter.acquire()
    assert limiter.remaining() == 0

    clock.now += 11  # 윈도우 밖으로 시간 이동
    assert limiter.remaining() == 1


def test_invalid_args_raise():
    import pytest

    with pytest.raises(ValueError):
        RateLimiter(0, 60)
    with pytest.raises(ValueError):
        RateLimiter(10, 0)
