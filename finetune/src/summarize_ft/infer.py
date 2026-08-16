"""학습된 LoRA adapter로 요약/인사이트 생성.

tools/summarize.py의 summarize_articles(), tools/insight.py의 extract_insights()와
동일한 입출력 시그니처를 맞춰뒀다 — 나중에 SUMMARIZER_PROVIDER=hf로 전환할 때
agents/summarizer.py에서 summarize_hf로 그대로 꽂아 넣을 수 있도록 하기 위함
(design doc 5절, agent-management-structure.md 참고).
"""
from __future__ import annotations

import json
from typing import Any

from .config import Config
from .prompt import build_insight_messages, build_summarize_messages
from .schema import SchemaError, validate_example


class GenerationError(RuntimeError):
    """모델 출력이 JSON으로 파싱되지 않을 때 발생."""


def load_adapter_for_inference(cfg: Config, adapter_path: str):
    """base model + adapter를 로드해 (model, tokenizer)를 반환한다. 추론 전용."""
    from peft import PeftModel

    from .modeling import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def _generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], *, max_new_tokens: int = 512) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 요약/인사이트는 재현성이 중요하므로 greedy decoding
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _parse_json_output(text: str) -> dict[str, Any]:
    text = text.strip()
    # tools/summarize.py, tools/insight.py와 동일하게 코드펜스가 섞여 나오는 경우를 방어한다.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"모델 출력이 JSON이 아님: {text[:200]!r}") from exc


def summarize_hf(articles: list[dict[str, Any]], model: Any, tokenizer: Any) -> list[dict[str, Any]]:
    """tools/summarize.py의 summarize_articles와 유사한 시그니처.

    주의(설계상 한계): tools/summarize.py는 여러 기사를 한 번에 묶어 여러 개의
    topic-summary로 클러스터링하는 방식이지만(1회 호출 -> N개 토픽), 이 학습
    파이프라인의 데이터는 sources/digests_export.py가 topic 단위로 explode해서
    "1개 topic(=근거 기사 묶음) -> 1개 요약" 형태로 만든다. 그래서 여기서는
    article 하나(혹은 이미 같은 topic으로 묶인 기사들)당 요약 1개를 생성하는
    방식으로 구현했다 — 실제 다건 기사를 자동으로 topic 클러스터링하려면
    tools/summarize.py처럼 별도 그룹핑 로직이 앞단에 더 필요하다.

    article dict는 raw_articles 스키마(title, description, url, source,
    published_at)를 따른다 — "content" 필드는 없음, description을 본문으로 쓴다.
    """
    results = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        article_text = f"제목: {title}\n설명: {description}"
        messages = build_summarize_messages(title, article_text)
        raw = _generate(model, tokenizer, messages)
        parsed = _parse_json_output(raw)
        parsed.setdefault("source_urls", [article["url"]] if article.get("url") else [])
        results.append(parsed)
    return results


def insight_hf(summaries: list[dict[str, Any]], model: Any, tokenizer: Any) -> dict[str, Any]:
    """tools/insight.py의 extract_insights와 동일한 시그니처."""
    messages = build_insight_messages(summaries)
    raw = _generate(model, tokenizer, messages)
    return _parse_json_output(raw)


def check_output_schema(task: str, source: str, input_: dict, output: dict, meta: dict) -> bool:
    """생성 결과가 evaluate.py 2단계(구조 검증)와 동일한 기준을 통과하는지 즉석 체크한다."""
    try:
        validate_example(
            {"id": "check", "task": task, "source": source, "input": input_, "output": output, "meta": meta}
        )
        return True
    except SchemaError:
        return False
