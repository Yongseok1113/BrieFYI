from datetime import datetime, timezone

from data_pipeline.sources import naver_source as naver_source_module
from data_pipeline.sources.naver_source import NaverNewsSource, _clean_text, _is_before_cutoff, _parse_pub_date


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _set_credentials(monkeypatch):
    monkeypatch.setattr(naver_source_module.config, "NAVER_CLIENT_ID", "test-id")
    monkeypatch.setattr(naver_source_module.config, "NAVER_CLIENT_SECRET", "test-secret")


def test_clean_text_strips_bold_tags_and_entities():
    assert _clean_text("삼성전자, <b>AI</b> 반도체 &quot;투자&quot; 확대") == '삼성전자, AI 반도체 "투자" 확대'


def test_parse_pub_date_converts_rfc822_to_iso8601():
    assert _parse_pub_date("Wed, 19 Aug 2026 10:30:00 +0900") == "2026-08-19T01:30:00Z"


def test_parse_pub_date_returns_none_for_invalid_input():
    assert _parse_pub_date("not a date") is None


def test_fetch_maps_naver_fields_to_common_schema(monkeypatch):
    _set_credentials(monkeypatch)
    payload = {
        "items": [
            {
                "title": "<b>AI</b> 투자 확대",
                "originallink": "https://example.com/original",
                "link": "https://news.naver.com/copy",
                "description": "삼성전자가 <b>AI</b> 반도체에 투자한다.",
                "pubDate": "Wed, 19 Aug 2026 10:30:00 +0900",
            }
        ]
    }
    monkeypatch.setattr(naver_source_module.requests, "get", lambda *a, **kw: FakeResponse(payload))

    articles = NaverNewsSource().fetch(keyword="AI", max_results=1)

    assert articles == [
        {
            "title": "AI 투자 확대",
            "description": "삼성전자가 AI 반도체에 투자한다.",
            "url": "https://example.com/original",
            "source": "네이버뉴스",
            "published_at": "2026-08-19T01:30:00Z",
        }
    ]


def test_fetch_falls_back_to_link_when_originallink_missing(monkeypatch):
    _set_credentials(monkeypatch)
    payload = {"items": [{"title": "t", "link": "https://news.naver.com/only", "description": "d", "pubDate": ""}]}
    monkeypatch.setattr(naver_source_module.requests, "get", lambda *a, **kw: FakeResponse(payload))

    articles = NaverNewsSource().fetch(keyword="AI", max_results=10)

    assert articles[0]["url"] == "https://news.naver.com/only"


def test_fetch_without_credentials_raises(monkeypatch):
    monkeypatch.setattr(naver_source_module.config, "NAVER_CLIENT_ID", "")
    monkeypatch.setattr(naver_source_module.config, "NAVER_CLIENT_SECRET", "")

    try:
        NaverNewsSource().fetch(keyword="AI")
        assert False, "RuntimeError를 기대했지만 발생하지 않음"
    except RuntimeError as exc:
        assert "NAVER_CLIENT_ID" in str(exc)


def test_fetch_paginates_until_max_results_reached(monkeypatch):
    _set_credentials(monkeypatch)
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(params.copy())
        start = params["start"]
        items = [
            {
                "title": f"기사{start + i}",
                "originallink": f"https://example.com/{start + i}",
                "description": "d",
                "pubDate": "Wed, 19 Aug 2026 10:30:00 +0900",
            }
            for i in range(params["display"])
        ]
        return FakeResponse({"items": items})

    monkeypatch.setattr(naver_source_module.requests, "get", fake_get)
    monkeypatch.setattr(naver_source_module, "_MAX_DISPLAY", 2)  # 페이지당 2건으로 축소해 페이징 유도

    articles = NaverNewsSource().fetch(keyword="AI", max_results=5)

    assert len(articles) == 5
    assert [c["start"] for c in calls] == [1, 3, 5]


def test_is_before_cutoff_compares_iso8601_dates():
    cutoff = datetime(2026, 7, 20, tzinfo=timezone.utc)
    assert _is_before_cutoff("2026-06-01T10:00:00Z", cutoff) is True
    assert _is_before_cutoff("2026-08-19T10:00:00Z", cutoff) is False


def test_is_before_cutoff_none_cutoff_or_date_never_filters():
    assert _is_before_cutoff("2026-06-01T10:00:00Z", None) is False
    assert _is_before_cutoff(None, datetime(2026, 7, 20, tzinfo=timezone.utc)) is False


def test_fetch_stops_at_lookback_cutoff(monkeypatch):
    _set_credentials(monkeypatch)
    payload = {
        "items": [
            {"title": "최신", "originallink": "https://example.com/new", "description": "d",
             "pubDate": "Wed, 19 Aug 2026 10:00:00 +0000"},
            {"title": "오래됨", "originallink": "https://example.com/old", "description": "d",
             "pubDate": "Mon, 01 Jun 2026 10:00:00 +0000"},
        ]
    }
    monkeypatch.setattr(naver_source_module.requests, "get", lambda *a, **kw: FakeResponse(payload))

    articles = NaverNewsSource().fetch(
        keyword="AI", lookback_days=30, max_results=10,
        now=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )

    assert [a["title"] for a in articles] == ["최신"]
