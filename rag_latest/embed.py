"""Hugging Face Inference API를 이용한 BGE-M3 dense embedding."""
import math

import requests

from config import config


def _endpoint(model_name: str) -> str:
    return (
        "https://router.huggingface.co/hf-inference/models/"
        f"{model_name}/pipeline/feature-extraction"
    )


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("Hugging Face가 유효하지 않은 embedding을 반환했습니다.")
    return [value / norm for value in vector]


def _validate_response(payload: object, expected_count: int) -> list[list[float]]:
    if isinstance(payload, dict):
        message = payload.get("error", "알 수 없는 응답")
        raise RuntimeError(f"Hugging Face embedding 실패: {message}")
    if not isinstance(payload, list):
        raise RuntimeError("Hugging Face embedding 응답이 배열이 아닙니다.")

    # 단일 문자열 요청에서 [dimension]으로 돌아오는 provider도 허용한다.
    if expected_count == 1 and payload and isinstance(payload[0], (int, float)):
        payload = [payload]

    if len(payload) != expected_count:
        raise RuntimeError(
            f"Hugging Face embedding 개수가 다릅니다: expected={expected_count}, actual={len(payload)}"
        )

    vectors: list[list[float]] = []
    for raw_vector in payload:
        if not isinstance(raw_vector, list) or len(raw_vector) != config.HF_EMBEDDING_DIMENSION:
            actual = len(raw_vector) if isinstance(raw_vector, list) else "not-array"
            raise RuntimeError(
                "Hugging Face embedding 차원이 다릅니다: "
                f"expected={config.HF_EMBEDDING_DIMENSION}, actual={actual}"
            )
        vector = [float(value) for value in raw_vector]
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("Hugging Face embedding에 NaN 또는 Infinity가 있습니다.")
        vectors.append(_normalize(vector))
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """문서 문자열 N개를 정규화된 [N, 1024] dense vector로 변환한다."""
    if not texts:
        return []
    if any(not text or not text.strip() for text in texts):
        raise ValueError("embedding 대상 텍스트는 비어 있을 수 없습니다.")
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN이 설정되지 않았습니다 (.env 확인)")

    response = requests.post(
        _endpoint(config.HF_EMBEDDING_MODEL),
        headers={"Authorization": f"Bearer {config.HF_TOKEN}"},
        json={"inputs": texts, "parameters": {"normalize": True}},
        timeout=config.HF_EMBEDDING_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:500]
        raise RuntimeError(
            f"Hugging Face embedding HTTP 오류: status={response.status_code}, body={detail}"
        ) from exc

    return _validate_response(response.json(), len(texts))


def embed_query(query: str) -> list[float]:
    """검색 query 하나를 문서와 같은 모델·정규화 방식의 [1024] vector로 변환한다."""
    if not query or not query.strip():
        raise ValueError("query는 비어 있을 수 없습니다.")
    return embed_texts([query.strip()])[0]
