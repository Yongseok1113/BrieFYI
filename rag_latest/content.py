"""기사 URL 본문 수집과 indexing 텍스트 구성, BGE token 청킹.

`requests` 외의 의존성(beautifulsoup4, transformers)은 실제 수집·청킹을 실행할 때만
import한다. 공유 requirements에는 등록하지 않았으므로 RAG worker 환경에 설치돼 있어야
하고, 없으면 lazy import 지점에서 필요한 package를 명시해 실패한다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import requests

from config import config

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


class ArticleContentDependencyError(RuntimeError):
    """본문 추출 실행 환경에 필요한 package가 없을 때 발생한다."""


# ---------------------------------------------------------------------------
# 본문 수집
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 텍스트 구성과 청킹
# ---------------------------------------------------------------------------

def build_article_text(
    title: str,
    description: str | None,
    body: str | None = None,
) -> str:
    """본문을 우선하고, 없으면 title+description indexing 텍스트를 만든다."""
    content = body if body and body.strip() else description
    parts = [part.strip() for part in (title, content or "") if part and part.strip()]
    return "\n\n".join(parts)


@lru_cache(maxsize=1)
def load_embedding_tokenizer():
    """현재 embedding model의 fast tokenizer를 한 번 load해 재사용한다."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "BGE token 청킹에는 transformers가 필요합니다. "
            "현재 RAG worker 환경에 설치해 주세요."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config.HF_EMBEDDING_MODEL,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"offset 청킹에는 fast tokenizer가 필요합니다: {config.HF_EMBEDDING_MODEL}"
        )
    return tokenizer


def split_text(
    text: str,
    tokenizer: Any,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """텍스트를 tokenizer의 offset mapping 기준으로 나눈다."""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap은 0 이상 chunk_size 미만이어야 합니다.")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = encoded["offset_mapping"]
    if not offsets:
        return []

    chunks: list[dict] = []
    token_start = 0
    while token_start < len(offsets):
        token_end = min(token_start + chunk_size, len(offsets))
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]
        chunks.append(
            {
                "chunk_index": len(chunks),
                "chunk_text": text[char_start:char_end],
            }
        )
        if token_end == len(offsets):
            break
        token_start = token_end - overlap

    return chunks


__all__ = [
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_SIZE_TOKENS",
    "ArticleContentDependencyError",
    "build_article_text",
    "fetch_article_body",
    "load_embedding_tokenizer",
    "split_text",
]
