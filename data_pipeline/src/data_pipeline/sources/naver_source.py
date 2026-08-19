"""네이버 뉴스 검색 API. GNewsSource와 동일한 구조화 스키마(title/description/url/source/
published_at)로 매핑해, ingest.py가 소스 종류를 신경 쓰지 않고 그대로 쓸 수 있게 한다.

네이버가 검색 API를 developers.naver.com(레거시)에서 NAVER API HUB(NAVER Cloud
Platform)로 이관하면서 엔드포인트와 인증 헤더가 바뀌었다 — NCP 콘솔에서 발급한
키(X-NCP-APIGW-API-KEY-ID/-KEY)는 레거시 openapi.naver.com 엔드포인트(X-Naver-Client-Id/
-Secret)에서는 인증되지 않는다. query/display/start/sort 파라미터와 응답 필드는 동일하다.
https://apihub.naver.com/devcenter/apps

GNews와의 차이:
  - 날짜 범위 파라미터가 없다(sort=date로 최신순만 가능) -- lookback_days가 오면
    클라이언트 쪽에서 pubDate 기준으로 걸러낸다.
  - 페이지당 최대 100건(display), start+display-1 <= 1000 제한이 있어 사실상 키워드당
    최대 1000건까지만 모을 수 있다 -- max_results가 이보다 크면 1000에서 멈춘다.
  - title/description에 HTML 태그(<b>)와 엔티티(&amp; 등)가 섞여 나와 정제가 필요하다.
  - pubDate가 RFC 822 형식이라 GNews와 맞추려면 ISO 8601로 변환해야 한다.
"""
from __future__ import annotations

import html
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from ..config import config
from .base import DataSource

_TAG_RE = re.compile(r"<[^>]+>")
_MAX_DISPLAY = 100
_MAX_START = 1000


def _clean_text(text: str) -> str:
    """<b> 하이라이트 태그와 HTML 엔티티(&quot; 등)를 제거한다."""
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


def _parse_pub_date(pub_date: str) -> str | None:
    """RFC 822(예: 'Tue, 19 Aug 2026 10:00:00 +0900')를 GNews와 같은 ISO 8601로 바꾼다."""
    try:
        dt = parsedate_to_datetime(pub_date)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_before_cutoff(published_at: str | None, cutoff: datetime | None) -> bool:
    """published_at(ISO 8601)이 cutoff보다 오래됐는지 판정한다. 순수 함수라 datetime.now()를
    몽키패치하지 않고도 테스트할 수 있다."""
    if cutoff is None or published_at is None:
        return False
    published_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return published_dt < cutoff


class NaverNewsSource(DataSource):
    name = "naver"
    is_structured = True

    def fetch(self, *, keyword: str | None = None, lookback_days: int | None = None,
              max_results: int | None = None, now: datetime | None = None, **_: Any) -> list[dict]:
        keyword = keyword or config.NEWS_KEYWORD
        max_results = max_results or config.NEWS_MAX_RESULTS
        if max_results > _MAX_START:
            max_results = _MAX_START

        if not config.NAVER_CLIENT_ID or not config.NAVER_CLIENT_SECRET:
            raise RuntimeError("NAVER_CLIENT_ID/NAVER_CLIENT_SECRET이 설정되지 않았습니다 (.env 확인)")

        cutoff = None
        if lookback_days is not None:
            cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=lookback_days)

        headers = {
            "X-NCP-APIGW-API-KEY-ID": config.NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": config.NAVER_CLIENT_SECRET,
        }

        articles: list[dict] = []
        start = 1
        while len(articles) < max_results and start <= _MAX_START:
            display = min(_MAX_DISPLAY, max_results - len(articles), _MAX_START - start + 1)
            params = {"query": keyword, "display": display, "start": start, "sort": "date"}

            for attempt in range(2):
                resp = requests.get(config.NAVER_NEWS_BASE_URL, params=params, headers=headers, timeout=30)
                if resp.status_code == 429 and attempt == 0:
                    time.sleep(1.0)
                    continue
                break
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                break

            reached_cutoff = False
            for item in items:
                published_at = _parse_pub_date(item.get("pubDate", ""))
                if _is_before_cutoff(published_at, cutoff):
                    reached_cutoff = True
                    continue
                articles.append(
                    {
                        "title": _clean_text(item.get("title", "")),
                        "description": _clean_text(item.get("description", "")),
                        "url": item.get("originallink") or item.get("link", ""),
                        "source": "네이버뉴스",
                        "published_at": published_at,
                    }
                )
                if len(articles) >= max_results:
                    break

            # sort=date라 최신순이므로, cutoff보다 오래된 기사가 나오기 시작하면 더 뒤져봐야 소용없다.
            if reached_cutoff:
                break
            start += display

        return articles[:max_results]
