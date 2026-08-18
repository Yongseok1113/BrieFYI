"""기사 텍스트 구성과 fixed-token 청킹."""
from typing import Any

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 0

# 전문 기사 corpus가 준비되면 아래 값으로 overlap을 활성화한다.
# CHUNK_OVERLAP_TOKENS = 50


def build_article_text(title: str, description: str | None) -> str:
    """현재 DB에 저장된 title과 description만으로 indexing 텍스트를 만든다."""
    parts = [part.strip() for part in (title, description or "") if part and part.strip()]
    return "\n\n".join(parts)


def split_text(
    text: str,
    tokenizer: Any | None = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """텍스트를 tokenizer offset 기준으로 나눈다.

    현재 HF API 방식은 로컬 tokenizer를 두지 않으므로 기사 전체를 청크 하나로
    반환한다. 향후 tokenizer를 전달하면 같은 인터페이스로 overlap 청킹을 쓸 수 있다.
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
