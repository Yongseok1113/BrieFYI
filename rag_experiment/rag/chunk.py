"""
RAG Chunking Module

기사 본문을 RAG 검색에 적합한 작은 텍스트 조각(chunk)으로 나눈다.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    긴 기사 본문을 여러 개의 chunk로 분할한다.

    Args:
        text: 분할할 기사 본문
        chunk_size: 하나의 chunk에 들어갈 최대 문자 수
        chunk_overlap: 이전 chunk와 겹치는 문자 수

    Returns:
        분할된 chunk 문자열 리스트
    """

    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    # 잘못된 설정 방지
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap은 chunk_size보다 작아야 합니다."
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # 다음 chunk 시작 위치
        start += chunk_size - chunk_overlap

    return chunks


if __name__ == "__main__":

    # 간단한 테스트
    sample_text = (
        "이것은 RAG chunking 테스트를 위한 예시 문장입니다. "
        "긴 뉴스 기사를 여러 개의 작은 문서로 나누어 "
        "검색할 수 있도록 만드는 것이 목적입니다. "
        "각 chunk는 일정한 길이를 가지며 일부 내용은 "
        "다음 chunk와 겹칠 수 있습니다."
    )

    chunks = chunk_text(
        sample_text,
        chunk_size=50,
        chunk_overlap=10
    )

    print("=" * 60)
    print("Chunking 테스트")
    print("=" * 60)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[Chunk {i}]")
        print(chunk)

    print("\n" + "=" * 60)
    print(f"전체 Chunk 수: {len(chunks)}")
    print("=" * 60)