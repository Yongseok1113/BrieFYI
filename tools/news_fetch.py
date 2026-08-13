"""뉴스 수집 도구 (구현 항목 #1). GNews Search Endpoint를 감싼다.
https://docs.gnews.io/endpoints/search-endpoint
"""
from datetime import datetime, timedelta, timezone

import requests

from config import config


def fetch_news(keyword: str, lookback_days: int = 1, max_results: int = 10) -> list[dict]:
    """키워드/기간으로 GNews를 검색해 정규화된 기사 리스트를 반환한다.

    반환 각 항목: {title, description, url, source, published_at}
    """
    if not config.GNEWS_API_KEY:
        raise RuntimeError("GNEWS_API_KEY가 설정되지 않았습니다 (.env 확인)")

    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=lookback_days)

    params = {
        "q": keyword,
        "lang": "ko",  # 필요 시 "en" 등으로 변경
        "max": max_results,
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sortby": "publishedAt",
        "apikey": config.GNEWS_API_KEY,
    }

    resp = requests.get(config.GNEWS_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for a in data.get("articles", []):
        articles.append(
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "content": a.get("content", ""),
                "url": a.get("url", ""),
                "source": (a.get("source") or {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
            }
        )
    return articles
