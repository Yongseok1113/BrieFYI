"""로컬 키워드 추출 (변형1). API 호출이 아니므로 요청 제한과 무관하고 비용도 0이다.

KeyBERT(다국어 문장 임베딩 기반)를 기본으로 쓰고, 모델 다운로드가 안 된 환경(테스트,
스모크 실행)에서는 빈도 기반 폴백으로 자동 전환한다 — 두 경로 모두 결과 형식은 동일하다.
"""
from __future__ import annotations

import re
from collections import Counter

from .config import config

_STOPWORDS = {
    "있다", "없다", "하다", "되다", "이다", "그", "이", "저", "것", "수", "등", "및", "위해",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
}

_keybert_model = None


def _extract_keybert(text: str, top_n: int) -> list[str]:
    global _keybert_model
    from keybert import KeyBERT

    if _keybert_model is None:
        _keybert_model = KeyBERT(model=config.KEYWORD_EMBEDDING_MODEL)

    pairs = _keybert_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=top_n
    )
    return [phrase for phrase, _score in pairs]


def _extract_simple(text: str, top_n: int) -> list[str]:
    """의존성 없는 빈도 기반 폴백. 한글/영문 2글자 이상 토큰만 대상으로 한다."""
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", text)
    tokens = [t.lower() if t.isascii() else t for t in tokens if t.lower() not in _STOPWORDS]
    counts = Counter(tokens)
    return [token for token, _count in counts.most_common(top_n)]


def extract_keywords(text: str, top_n: int | None = None, *, use_keybert: bool = True) -> list[str]:
    top_n = top_n or config.KEYWORD_TOP_N
    if not text or not text.strip():
        return []

    if use_keybert:
        try:
            return _extract_keybert(text, top_n)
        except ImportError:
            pass  # keybert/sentence-transformers 미설치 -> 폴백
    return _extract_simple(text, top_n)
