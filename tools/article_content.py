"""기사 URL에서 검색·인덱싱용 본문을 가져오는 도구.

BeautifulSoup은 실제 본문 수집을 실행할 때만 import한다. 현재 requirements에는
의존성을 추가하지 않았으므로, worker 환경에 beautifulsoup4가 설치되어 있어야 한다.
"""
from __future__ import annotations

import requests


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


class ArticleContentDependencyError(RuntimeError):
    """본문 추출 실행 환경에 필요한 package가 없을 때 발생한다."""


def _load_beautiful_soup():
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ArticleContentDependencyError(
            "기사 본문 수집에는 beautifulsoup4가 필요합니다. "
            "현재 worker 환경에 설치해 주세요."
        ) from exc
    return BeautifulSoup


def _paragraphs(container, *, min_chars: int = 0) -> list[str]:
    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in container.find_all("p")
    ]
    return [text for text in paragraphs if text and len(text) > min_chars]


def fetch_article_body(url: str, timeout: int = 15) -> str:
    """기사 URL의 HTML에서 best-effort로 본문을 추출한다.

    본문을 얻지 못하면 빈 문자열을 반환하지 않고 예외를 발생시킨다. 호출자는 이
    경계를 기준으로 title+description fallback 여부를 결정할 수 있다.
    """
    url = url.strip()
    if not url:
        raise ValueError("기사 URL은 비어 있을 수 없습니다.")
    if timeout <= 0:
        raise ValueError("timeout은 0보다 커야 합니다.")

    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and "html" not in content_type:
        raise ValueError(f"HTML 문서가 아닙니다: content_type={content_type}")

    BeautifulSoup = _load_beautiful_soup()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for selector in ("article", "main", '[role="main"]'):
        container = soup.select_one(selector)
        if container is None:
            continue
        paragraphs = _paragraphs(container)
        if paragraphs:
            return "\n\n".join(paragraphs)
        text = container.get_text(" ", strip=True)
        if text:
            return text

    # 문서 전체를 보는 마지막 fallback에서는 짧은 메뉴·캡션 문단을 제외한다.
    paragraphs = _paragraphs(soup, min_chars=40)
    if paragraphs:
        return "\n\n".join(paragraphs)
    raise ValueError("기사 본문을 추출하지 못했습니다.")


__all__ = ["ArticleContentDependencyError", "fetch_article_body"]
