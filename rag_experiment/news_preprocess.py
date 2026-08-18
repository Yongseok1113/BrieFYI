import re


# ==========================================
# 1. 뉴스 원문 전처리
# ==========================================

def clean_text(text: str) -> str:
    """
    뉴스 원문에서 불필요한 공백과 줄바꿈을 정리한다.
    """

    if not text:
        return ""

    # 줄바꿈을 공백으로 변경
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # 여러 개의 공백을 하나로
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 2. 뉴스 원문 청킹
# ==========================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:
    """
    뉴스 원문을 문장 단위로 청킹한다.

    - 문장 중간에서 자르지 않는다.
    - 하나의 청크가 최대 chunk_size에 가깝도록 문장을 묶는다.
    - 청크가 넘어갈 때 이전 문장을 일부 유지하여 문맥을 이어준다.
    """

    text = clean_text(text)

    if not text:
        return []

    # ==========================================
    # 문장 단위 분리
    # ==========================================

    sentences = re.split(
        r"(?<=[.!?。！？])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    current_sentences = []
    current_length = 0

    # ==========================================
    # 청크 생성
    # ==========================================

    for sentence in sentences:

        sentence_length = len(sentence)

        # 현재 청크에 문장을 추가했을 때 길이
        new_length = (
            current_length
            + sentence_length
            + 1
        )

        # ======================================
        # chunk_size를 초과하면 현재 청크 저장
        # ======================================

        if (
            current_sentences
            and new_length > chunk_size
        ):

            chunk = " ".join(current_sentences)

            chunks.append(chunk)

            # ==================================
            # overlap 처리
            # ==================================

            overlap_sentences = []
            overlap_length = 0

            # 현재 청크의 뒤쪽 문장부터 확인
            for previous_sentence in reversed(
                current_sentences
            ):

                previous_length = len(
                    previous_sentence
                )

                # overlap 범위 안에 들어오는
                # 마지막 문장을 유지
                if (
                    overlap_length
                    + previous_length
                    + 1
                    <= chunk_overlap
                ):

                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_length += (
                        previous_length + 1
                    )

                else:
                    break

            # 다음 청크는 overlap 문장으로 시작
            current_sentences = overlap_sentences
            current_length = overlap_length

        # ======================================
        # 현재 문장을 청크에 추가
        # ======================================

        current_sentences.append(sentence)

        current_length += sentence_length + 1

    # ==========================================
    # 마지막 청크 저장
    # ==========================================

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


# ==========================================
# 3. 테스트
# ==========================================

if __name__ == "__main__":

    test_text = """
    정부가 오늘 일본 정계의 야스쿠니 신사 공물 봉납과
    참배를 비판했습니다.

    외교부는 오늘 대변인 명의의 논평에서
    일본의 책임 있는 지도자들이 역사를 직시하고
    과거사에 대한 겸허한 성찰과 진정한 반성을
    행동으로 보여줄 것을 촉구했습니다.

    정부는 앞으로도 실용적 국익 외교를 바탕으로
    공동의 이익을 위한 협력을 확대하겠다고 밝혔습니다.
    """

    print("=" * 60)
    print("뉴스 전처리 테스트")
    print("=" * 60)

    # ==========================================
    # 전처리
    # ==========================================

    cleaned = clean_text(test_text)

    print("\n[전처리 결과]")
    print(cleaned)

    # ==========================================
    # 청킹
    # ==========================================

    chunks = chunk_text(
        cleaned,
        chunk_size=500,
        chunk_overlap=50
    )

    print("\n" + "=" * 60)
    print(f"생성된 청크: {len(chunks)}개")
    print("=" * 60)

    # ==========================================
    # 결과 출력
    # ==========================================

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print(f"\n[청크 {i}]")
        print(f"길이: {len(chunk)}자")
        print(chunk)