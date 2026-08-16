"""Hugging Face Inference API 클라이언트 공용 헬퍼.

finetune/으로 학습해 Hub에 병합·업로드한 요약 모델을, 로컬 GPU 없이 Claude처럼
"모델명 하나로" 호출하기 위한 용도다 (SUMMARIZER_PROVIDER=hf).
tools/llm_client.py(Anthropic)와 동일한 인터페이스(call_llm/parse_json_response)를
맞춰뒀기 때문에 tools/summarize_hf.py 쪽에서 거의 그대로 재사용할 수 있다.

HF_MODEL_ID는 코랩 노트북(finetune/notebooks/qlora_qwen3_8b_colab.ipynb, 12절)에서
병합 후 push한 모델 저장소 이름이다 (예: "username/briefyi-qwen3-8b-summarize").
"""
from huggingface_hub import InferenceClient

from config import config

from .llm_client import parse_json_response  # noqa: F401  (같은 파싱 로직을 재사용하기 위한 재노출)

_client: InferenceClient | None = None


def get_client() -> InferenceClient:
    global _client
    if _client is None:
        if not config.HF_API_TOKEN:
            raise RuntimeError("HF_API_TOKEN이 설정되지 않았습니다 (.env 확인)")
        if not config.HF_MODEL_ID:
            raise RuntimeError(
                "HF_MODEL_ID가 설정되지 않았습니다 (.env 확인, 예: username/briefyi-qwen3-8b-summarize)"
            )
        _client = InferenceClient(model=config.HF_MODEL_ID, token=config.HF_API_TOKEN)
    return _client


def call_llm(system: str, user: str, max_tokens: int = 2000) -> str:
    """tools/llm_client.py의 call_llm과 동일한 시그니처.

    chat_completion은 OpenAI 호환 채팅 API라 프롬프트를 system/user 메시지로
    그대로 넘길 수 있다 — Qwen3 계열은 chat template이 system role을 지원한다.

    주의: HF_MODEL_ID가 무료 서버리스 Inference API에서 자동으로 서빙되지 않는
    커스텀 파인튜닝 모델이면(콜드스타트 모델이 아니면) 호출이 실패할 수 있다.
    이 경우 Hugging Face Inference Endpoints(유료, 전용 GPU)를 만들고
    InferenceClient(model=<endpoint_url>, token=...) 형태로 endpoint URL을
    넘기도록 바꾸면 된다 — call_llm 인터페이스 자체는 바뀌지 않는다.
    """
    client = get_client()
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
