"""GLiNER2로 기사 metadata(Category/Domain/Entity)와 구조화 Event를 추출한다.

두 추출기는 같은 GLiNER2 model 하나를 공유하므로 한 파일에 둔다. 모델 import와 load는
`load_gliner2_model()` 호출 시점까지 미루므로, 추출을 실제로 실행하지 않는 일반 RAG
경로와 unit test는 GLiNER2, torch, 모델 다운로드를 요구하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .taxonomy import (
    CATEGORY_LABELS,
    DOMAIN_CLASSIFICATION_THRESHOLD,
    DOMAIN_LABELS,
    ENTITY_TYPES,
    GLINER2_MODEL_NAME,
    MAX_ENTITIES,
    MAX_TOKENS_PER_WINDOW,
    OVERLAP_WORDS,
    RELATION_ROLES,
    RELATION_THRESHOLD,
    SYMMETRIC_RELATIONS,
)

KOREAN_PARTICLES = (
    "에게서",
    "으로부터",
    "에서",
    "에게",
    "께서",
    "으로",
    "부터",
    "까지",
    "이나",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "와",
    "과",
    "도",
    "만",
    "로",
    "나",
    "에",
)


# ---------------------------------------------------------------------------
# 공통: 텍스트 정규화와 window 분할
# ---------------------------------------------------------------------------

def normalize_argument_text(text: str) -> str:
    """공백과 흔한 한국어 조사를 정리하되 원문 text 자체는 별도로 보존한다."""
    normalized = " ".join(text.split())
    for particle in sorted(KOREAN_PARTICLES, key=len, reverse=True):
        if normalized.endswith(particle) and len(normalized) > len(particle):
            return normalized[: -len(particle)].rstrip()
    return normalized


def _normalized_key(text: str) -> str:
    return normalize_argument_text(text).casefold()


def split_text_windows(
    text: str,
    tokenizer: Any,
    *,
    max_tokens: int = MAX_TOKENS_PER_WINDOW,
    overlap_words: int = OVERLAP_WORDS,
) -> list[dict]:
    """실제 tokenizer 토큰 수로 나누고 부모 chunk의 문자 offset을 보존한다."""
    if max_tokens <= 0:
        raise ValueError("max_tokens는 1 이상이어야 합니다.")
    if overlap_words < 0:
        raise ValueError("overlap_words는 0 이상이어야 합니다.")

    words = list(re.finditer(r"\S+", text))
    if not words:
        return []

    windows: list[dict] = []
    start_word = 0
    while start_word < len(words):
        char_start = words[start_word].start()
        end_word = start_word + 1

        while end_word < len(words):
            candidate_end = words[end_word].end()
            candidate = text[char_start:candidate_end]
            token_count = len(tokenizer.encode(candidate, add_special_tokens=False))
            if token_count > max_tokens:
                break
            end_word += 1

        char_end = words[end_word - 1].end()
        windows.append(
            {
                "text": text[char_start:char_end],
                "start": char_start,
                "end": char_end,
            }
        )
        if end_word >= len(words):
            break
        start_word = max(start_word + 1, end_word - overlap_words)

    return windows


# ---------------------------------------------------------------------------
# 구조화 Event
# ---------------------------------------------------------------------------

def _entity_keys(result: dict) -> set[str]:
    keys: set[str] = set()
    entities = result.get("entities", {})
    if not isinstance(entities, dict):
        return keys

    for items in entities.values():
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            text = item.get("text") if isinstance(item, dict) else item
            if isinstance(text, str) and _normalized_key(text):
                keys.add(_normalized_key(text))
    return keys


def _argument_from_result(
    item: object,
    *,
    role: str,
    source_text: str,
    source_offset: int,
) -> dict | None:
    if not isinstance(item, dict):
        return None
    start = item.get("start")
    end = item.get("end")
    confidence = item.get("confidence")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if not (0 <= start < end <= len(source_text)):
        return None
    if not isinstance(confidence, (int, float)):
        return None

    raw_text = source_text[start:end]
    normalized_text = normalize_argument_text(raw_text)
    if not normalized_text:
        return None
    return {
        "role": role,
        "text": raw_text,
        "normalized_text": normalized_text,
        "confidence": float(confidence),
        "span_start": source_offset + start,
        "span_end": source_offset + end,
    }


def event_candidates_from_result(
    result: dict,
    source_text: str,
    *,
    source_offset: int = 0,
) -> list[dict]:
    """한 GLiNER2 window 결과를 검증해 구조화 Event 후보로 바꾼다."""
    entity_keys = _entity_keys(result)
    relations = result.get("relation_extraction", {})
    if not entity_keys or not isinstance(relations, dict):
        return []

    candidates: list[dict] = []
    for event_type, (head_role, tail_role) in RELATION_ROLES.items():
        instances = relations.get(event_type, [])
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            head = _argument_from_result(
                instance.get("head"),
                role=head_role,
                source_text=source_text,
                source_offset=source_offset,
            )
            tail = _argument_from_result(
                instance.get("tail"),
                role=tail_role,
                source_text=source_text,
                source_offset=source_offset,
            )
            if head is None or tail is None:
                continue

            head_key = head["normalized_text"].casefold()
            tail_key = tail["normalized_text"].casefold()
            if head_key == tail_key:
                continue
            if head_key not in entity_keys or tail_key not in entity_keys:
                continue

            confidence = min(head["confidence"], tail["confidence"])
            if confidence < RELATION_THRESHOLD:
                continue

            arguments = [head, tail]
            if event_type in SYMMETRIC_RELATIONS:
                arguments.sort(key=lambda arg: arg["normalized_text"].casefold())
                arguments[0]["role"], arguments[1]["role"] = head_role, tail_role

            candidates.append(
                {
                    "event_type": event_type,
                    "confidence": confidence,
                    "evidence_start": min(arg["span_start"] for arg in arguments),
                    "evidence_end": max(arg["span_end"] for arg in arguments),
                    "arguments": arguments,
                }
            )
    return candidates


def _pair_key(event: dict) -> tuple[str, str]:
    return tuple(
        argument["normalized_text"].casefold()
        for argument in event["arguments"]
    )


def _event_fingerprint(event: dict) -> str:
    payload = {
        "event_type": event["event_type"],
        "arguments": [
            {
                "role": argument["role"],
                "text": argument["normalized_text"].casefold(),
            }
            for argument in event["arguments"]
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_event_candidates(candidates: list[dict]) -> list[dict]:
    """argument 쌍별 최고 relation을 고르고 overlap/여러 chunk 중복을 제거한다."""
    best_by_pair: dict[tuple[str, str], dict] = {}
    for candidate in candidates:
        pair = _pair_key(candidate)
        previous = best_by_pair.get(pair)
        if previous is None or candidate["confidence"] > previous["confidence"]:
            best_by_pair[pair] = candidate

    best_by_fingerprint: dict[str, dict] = {}
    for candidate in best_by_pair.values():
        fingerprint = _event_fingerprint(candidate)
        candidate = {**candidate, "event_fingerprint": fingerprint}
        previous = best_by_fingerprint.get(fingerprint)
        if previous is None or candidate["confidence"] > previous["confidence"]:
            best_by_fingerprint[fingerprint] = candidate

    return sorted(
        best_by_fingerprint.values(),
        key=lambda event: (-event["confidence"], event["event_type"], event["event_fingerprint"]),
    )


class GLiNER2EventExtractor:
    """로드된 GLiNER2 모델 하나를 여러 article chunk에 재사용한다."""

    def __init__(self, model: Any):
        self.model = model
        self.tokenizer = model.processor.tokenizer
        # GLiNER2 1.3.2 multi-v1은 Entity와 Relation을 한 schema로 합치면 같은 문장에서
        # relation은 반환하면서 entities가 빈 dict가 되는 경우가 확인됐다. 같은 window를
        # 두 task schema로 각각 실행한 뒤 합쳐 Entity 일치 검증을 수행한다.
        self.entity_schema = model.create_schema().entities(ENTITY_TYPES)
        self.relation_schema = model.create_schema().relations(
            list(RELATION_ROLES),
            threshold=RELATION_THRESHOLD,
        )

    def extract(self, text: str) -> list[dict]:
        candidates: list[dict] = []
        for window in split_text_windows(text, self.tokenizer):
            entity_result = self.model.extract(
                window["text"],
                self.entity_schema,
                threshold=RELATION_THRESHOLD,
                include_confidence=True,
                include_spans=True,
            )
            relation_result = self.model.extract(
                window["text"],
                self.relation_schema,
                threshold=RELATION_THRESHOLD,
                include_confidence=True,
                include_spans=True,
            )
            result = {
                "entities": entity_result.get("entities", {}),
                "relation_extraction": relation_result.get("relation_extraction", {}),
            }
            candidates.extend(
                event_candidates_from_result(
                    result,
                    window["text"],
                    source_offset=window["start"],
                )
            )
        return aggregate_event_candidates(candidates)


# ---------------------------------------------------------------------------
# Category / Domain / Entity
# ---------------------------------------------------------------------------

class GLiNER2TopicExtractor:
    """로드된 GLiNER2 model을 재사용해 article-level metadata를 집계한다."""

    def __init__(self, model: Any):
        self.model = model
        self.tokenizer = model.processor.tokenizer
        self.schema = (
            model.create_schema()
            .entities(ENTITY_TYPES)
            .classification("category", list(CATEGORY_LABELS))
            .classification(
                "domain",
                list(DOMAIN_LABELS),
                multi_label=True,
                cls_threshold=DOMAIN_CLASSIFICATION_THRESHOLD,
            )
        )

    def extract(self, text: str) -> dict:
        """모든 GLiNER window 결과를 article-level metadata로 합친다."""
        windows = split_text_windows(text, self.tokenizer)
        if not windows:
            raise ValueError("topic extraction 텍스트가 비어 있습니다.")

        results = [
            self.model.extract(window["text"], self.schema, include_confidence=True)
            for window in windows
        ]

        categories = [
            result.get("category")
            for result in results
            if isinstance(result.get("category"), dict)
            and result["category"].get("label") in CATEGORY_LABELS
        ]
        best_category = (
            max(categories, key=lambda item: float(item.get("confidence", 0.0)))["label"]
            if categories
            else "기타"
        )

        domain_scores: dict[str, float] = {}
        for result in results:
            domains = result.get("domain", [])
            if isinstance(domains, dict):
                domains = [domains]
            if not isinstance(domains, list):
                continue
            for item in domains:
                if not isinstance(item, dict) or item.get("label") not in DOMAIN_LABELS:
                    continue
                label = item["label"]
                domain_scores[label] = max(
                    domain_scores.get(label, 0.0),
                    float(item.get("confidence", 0.0)),
                )
        if len(domain_scores) > 1:
            domain_scores.pop("기타", None)
        domains = [
            label
            for label, _score in sorted(
                domain_scores.items(),
                key=lambda item: (-item[1], DOMAIN_LABELS.index(item[0])),
            )
        ] or ["기타"]

        entities: dict[str, tuple[str, float]] = {}
        for result in results:
            entity_groups = result.get("entities", {})
            if not isinstance(entity_groups, dict):
                continue
            for items in entity_groups.values():
                if isinstance(items, dict):
                    items = [items]
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                        continue
                    text = normalize_argument_text(item["text"])
                    if not text:
                        continue
                    key = text.casefold()
                    confidence = float(item.get("confidence", 0.0))
                    if key not in entities or confidence > entities[key][1]:
                        entities[key] = (text, confidence)
        top_entities = [
            text
            for text, _confidence in sorted(
                entities.values(),
                key=lambda item: (-item[1], item[0].casefold()),
            )[:MAX_ENTITIES]
        ]

        return {
            "category": best_category,
            "domains": domains,
            "entities": top_entities,
        }


# ---------------------------------------------------------------------------
# 모델 load
# ---------------------------------------------------------------------------

def load_gliner2_model(device: str = "cuda") -> Any:
    """요청한 device에 GLiNER2를 load한다. CUDA 요청을 CPU로 자동 전환하지 않는다."""
    if device not in {"cuda", "cpu"}:
        raise ValueError("device는 'cuda' 또는 'cpu'여야 합니다.")

    try:
        import torch
        from gliner2 import GLiNER2
    except ImportError as exc:
        raise RuntimeError(
            "GLiNER2 실행 의존성이 없습니다. "
            "rag/requirements-event.txt를 local 환경에 설치해 주세요."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 PyTorch에서 CUDA를 사용할 수 없습니다.")

    return GLiNER2.from_pretrained(GLINER2_MODEL_NAME, map_location=device)


def load_event_extractor(device: str = "cuda") -> GLiNER2EventExtractor:
    """Event만 단독으로 인덱싱할 때 쓰는 편의 loader."""
    return GLiNER2EventExtractor(load_gliner2_model(device))


__all__ = [
    "GLiNER2EventExtractor",
    "GLiNER2TopicExtractor",
    "aggregate_event_candidates",
    "event_candidates_from_result",
    "load_event_extractor",
    "load_gliner2_model",
    "normalize_argument_text",
    "split_text_windows",
]
