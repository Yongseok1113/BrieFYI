"""기사 1건 -> (entities, category, domains). enrichment가 있으면 그대로 쓰고,
없으면 시드 엔티티+별칭사전(원 설계문서 2.2절 목록 기반)을 title+description에
문자열 매칭한다 — 새 NER 의존성을 추가하지 않는다.
"""
from __future__ import annotations

# alias -> canonical. 원 설계문서 2.2절의 시드 엔티티(빅테크 + 확장 후보)를 기반으로 한다.
SEED_ENTITY_ALIASES: dict[str, str] = {
    "NVIDIA": "NVIDIA", "엔비디아": "NVIDIA",
    "OpenAI": "OpenAI", "오픈AI": "OpenAI", "오픈에이아이": "OpenAI",
    "Anthropic": "Anthropic", "앤트로픽": "Anthropic",
    "Google DeepMind": "Google DeepMind",
    "Google": "Google", "구글": "Google",
    "Microsoft": "Microsoft", "마이크로소프트": "Microsoft",
    "Meta AI": "Meta", "Meta": "Meta", "메타": "Meta",
    "TensorFlow": "TensorFlow",
    "PyTorch": "PyTorch",
    "TSMC": "TSMC",
    "ASML": "ASML",
    "AMD": "AMD",
    "xAI": "xAI",
    "Mistral": "Mistral",
    "Hugging Face": "Hugging Face",
    "AWS": "Amazon", "Amazon": "Amazon", "아마존": "Amazon",
    "Samsung": "Samsung", "삼성전자": "Samsung", "삼성": "Samsung",
}

# 긴 별칭부터 매칭해야 "Google"이 "Google DeepMind"를 가리는 걸 방지한다.
_ALIASES_BY_LENGTH = sorted(SEED_ENTITY_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)


def extract(article: dict) -> tuple[list[str], str | None, list[str]]:
    if article.get("entity"):
        entities = list(article["entity"])
        category = article.get("category")
        domains = list(article.get("domain") or [])
        return entities, category, domains

    text = f"{article.get('title') or ''} {article.get('description') or ''}"
    found: list[str] = []
    remaining = text
    for alias, canonical in _ALIASES_BY_LENGTH:
        if alias in remaining and canonical not in found:
            found.append(canonical)
    return sorted(found), None, []
