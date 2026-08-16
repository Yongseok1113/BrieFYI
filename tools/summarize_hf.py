"""tools/summarize.py의 HF 파인튜닝 모델 버전.

프롬프트/출력 파싱은 tools/summarize.py와 완전히 동일하게 맞췄다 — 학습 데이터
자체가 이 프롬프트로 생성된 Claude 출력을 교사 데이터로 삼았기 때문에(distillation),
프롬프트를 다르게 주면 학습 때 본 분포와 어긋나 품질이 떨어진다.
클라이언트만 tools/llm_client.py(Anthropic) 대신 tools/hf_llm_client.py(HF Inference API)로 바뀐다.

agents/registry.py에서 SUMMARIZER_PROVIDER=hf일 때 이 함수가 "summarize_hf" tool로 등록된다.
"""
from .hf_llm_client import call_llm, parse_json_response
from .summarize import SYSTEM_PROMPT


def summarize_articles_hf(articles: list[dict]) -> list[dict]:
    """원시 기사 리스트 -> 주제별 요약 리스트. summarize_articles()와 동일한 시그니처."""
    if not articles:
        return []

    article_text = "\n\n".join(
        f"제목: {a['title']}\n설명: {a.get('description', '')}\nURL: {a['url']}"
        for a in articles
    )
    user_prompt = f"다음 기사들을 요약해줘:\n\n{article_text}"

    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    return parse_json_response(raw)
