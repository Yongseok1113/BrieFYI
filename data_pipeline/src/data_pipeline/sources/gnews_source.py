"""GNews Search API. 구조화 소스라 LLM 없이 필드를 그대로 매핑한다 (tools/news_fetch.py와 동일 API,
data_pipeline은 별도 컨테이너라 자체 구현을 둔다).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ..config import config
from .base import DataSource


class GNewsSource(DataSource):
    name = "gnews"
    is_structured = True

    def fetch(self, *, keyword: str | None = None, lookback_days: int | None = None,
              max_results: int | None = None, **_: Any) -> list[dict]:
        keyword = keyword or config.NEWS_KEYWORD
        lookback_days = lookback_days or config.NEWS_LOOKBACK_DAYS
        max_results = max_results or config.NEWS_MAX_RESULTS

        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "q": keyword,
            "lang": "ko",
            "max": max_results,
            "from": since,
            "sortby": "publishedAt",
            "apikey": config.GNEWS_API_KEY,
        }
        resp = requests.get(config.GNEWS_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "source": (a.get("source") or {}).get("name", ""),
                "published_at": a.get("publishedAt"),
            }
            for a in articles
        ]
