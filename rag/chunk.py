"""기사 텍스트 구성과 fixed-token 청킹.

실제 BGE-M3 tokenizer 청킹에는 ``transformers.AutoTokenizer``가 필요하다.
requirements 등록은 아직 보류했으므로, RAG worker를 실행하는 환경에 transformers가
설치되어 있어야 한다. import와 model load는 worker 실행 시점까지 미룬다.
"""
from functools import lru_cache
from typing import Any

from config import config

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


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
    tokenizer: Any | None = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """텍스트를 tokenizer offset 기준으로 나눈다.

    tokenizer가 없으면 짧은 helper 사용을 위해 전체 텍스트를 한 chunk로 반환한다.
    실제 indexer는 ``load_embedding_tokenizer()`` 결과를 항상 전달한다.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap은 0 이상 chunk_size 미만이어야 합니다.")

    if tokenizer is None:
        return [{"chunk_index": 0, "chunk_text": text}]

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
    "build_article_text",
    "load_embedding_tokenizer",
    "split_text",
]
