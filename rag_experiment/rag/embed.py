"""
RAG Embedding Module

텍스트를 임베딩 벡터로 변환한다.
"""

from typing import List

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"

model = None


def get_model():
    """
    임베딩 모델이 필요할 때 한 번만 로딩한다.
    """

    global model

    if model is None:
        print("\n" + "=" * 60)
        print("Embedding 모델 로딩 중...")
        print(f"모델: {MODEL_NAME}")
        print("=" * 60)

        model = SentenceTransformer(MODEL_NAME)

        print("Embedding 모델 로딩 완료")

    return model


def embed_text(text: str) -> List[float]:
    """
    하나의 텍스트를 임베딩 벡터로 변환한다.
    """

    if not text or not text.strip():
        raise ValueError("임베딩할 텍스트가 없습니다.")

    embedding_model = get_model()

    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


def embed_texts(
    texts: List[str]
) -> List[List[float]]:
    """
    여러 텍스트를 임베딩 벡터로 변환한다.
    """

    if not texts:
        return []

    embedding_model = get_model()

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()


if __name__ == "__main__":

    sample_texts = [
        "프로야구 경기에서 새로운 기록이 나왔다.",
        "인공지능 기술이 다양한 산업에서 활용되고 있다."
    ]

    embeddings = embed_texts(sample_texts)

    print("=" * 60)
    print("Embedding 테스트")
    print("=" * 60)

    for index, embedding in enumerate(
        embeddings,
        start=1
    ):

        print(f"\n[텍스트 {index}]")
        print(sample_texts[index - 1])
        print(f"벡터 차원: {len(embedding)}")
        print(f"벡터 앞부분: {embedding[:5]}")

    print("\n" + "=" * 60)
    print("Embedding 테스트 완료")
    print("=" * 60)