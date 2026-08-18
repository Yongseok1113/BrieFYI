"""
RAG Embedding Module

텍스트를 임베딩 벡터로 변환한다.
"""

from typing import List

# gc: 파이썬의 "가비지 컬렉터"를 직접 다루는 기본 내장 라이브러리.
# 더 이상 안 쓰는 객체(메모리)를 강제로 정리해달라고 요청할 때 쓴다.
import gc

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


def unload_model():
    """
    로딩해둔 임베딩 모델을 메모리에서 내려놓는다.

    사용 시점:
        검색어 임베딩(retrieve_documents)까지 다 끝내고,
        이제 리랭커 모델을 새로 로딩해야 하는 직전 시점에 호출한다.

    동작 방식:
        global model
            → 이 함수 안에서 맨 위에 있는 전역 변수 model을
              직접 바꾸겠다는 선언. 이게 없으면 파이썬은
              이 함수 안에서만 쓰는 새로운 지역 변수로 착각한다.

        model = None
            → 지금까지 model 변수가 가리키고 있던
              거대한 모델 데이터에 대한 "참조"를 끊어버린다.
              더 이상 아무도 그 데이터를 안 가리키게 되면
              파이썬이 알아서 메모리에서 지운다.

        gc.collect()
            → "지금 당장 안 쓰는 메모리 있으면 싹 정리해줘"라고
              파이썬에게 강제로 요청하는 것.
              평소엔 파이썬이 알아서 시점을 정해서 정리하는데,
              지금처럼 "바로 다음 줄에서 큰 모델을 또 로딩해야 하니
              지금 당장 비워달라"고 확실히 하고 싶을 때 직접 호출한다.
    """

    global model

    if model is not None:

        print("\nEmbedding 모델 메모리에서 해제 중...")

        model = None

        gc.collect()

        print("Embedding 모델 해제 완료")


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