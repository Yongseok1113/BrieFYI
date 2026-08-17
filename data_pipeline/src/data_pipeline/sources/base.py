"""데이터 소스 공통 인터페이스.

is_structured=True인 소스는 ingest.py가 LLM 없이 필드를 그대로 매핑한다.
is_structured=False인 소스(향후 스크래핑 등)는 raw_content를 LLM으로 정제해야 한다 —
파이프라인 파라미터(--source, is_structured)로 이 분기를 제어한다 (design doc 2절).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataSource(ABC):
    name: str
    is_structured: bool

    @abstractmethod
    def fetch(self, **kwargs: Any) -> list[dict]:
        """소스별 원시 데이터를 가져온다.

        is_structured=True: 반환 dict는 이미 title/description/url/source/published_at을 포함.
        is_structured=False: 반환 dict는 raw_content(정제 전 원문) 등 임의 필드를 포함해도 되고,
        ingest.py가 llm_client로 구조화 필드를 뽑아낸다.
        """
        raise NotImplementedError
