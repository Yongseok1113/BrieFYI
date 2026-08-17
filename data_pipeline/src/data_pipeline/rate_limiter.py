"""요청 제한을 지키기 위한 슬라이딩 윈도우 레이트 리미터 (design doc 7절).

모든 LLM 호출(llm_client.py)이 이 리미터를 거쳐 나간다. HF 무료 티어처럼
시간당 요청 수가 제한된 API를 안전하게 쓰기 위한 용도다.
"""
from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float, *, sleep_fn=time.sleep,
                 time_fn=time.monotonic):
        if max_requests <= 0:
            raise ValueError("max_requests는 양수여야 함")
        if window_seconds <= 0:
            raise ValueError("window_seconds는 양수여야 함")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._sleep = sleep_fn
        self._now = time_fn

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        # <= 를 써야 한다: acquire()의 wait 계산은 `timestamps[0] <= cutoff`일 때 대기 시간을
        # 0으로 판단한다. 여기서 엄격한 `<`를 쓰면 딱 경계값(timestamps[0] == cutoff)에서
        # "만료도 아니고 더 잘 필요도 없는" 상태가 돼 acquire()가 그대로 무한 루프에 빠진다.
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def acquire(self) -> None:
        """윈도우 안에 여유가 생길 때까지 블로킹하고, 여유가 생기면 요청 1건을 기록한다."""
        while True:
            now = self._now()
            self._prune(now)
            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return
            wait = self._timestamps[0] + self.window_seconds - now
            if wait > 0:
                self._sleep(wait)

    def remaining(self) -> int:
        self._prune(self._now())
        return max(0, self.max_requests - len(self._timestamps))
