"""GLiNER2로 기사 Category/Domain/Entity를 자동 추출한다."""
from __future__ import annotations

from typing import Any

from .event_extractor import normalize_argument_text, split_text_windows
from .event_taxonomy import ENTITY_TYPES
from .topic_taxonomy import (
    CATEGORY_LABELS,
    DOMAIN_CLASSIFICATION_THRESHOLD,
    DOMAIN_LABELS,
    MAX_ENTITIES,
    TOPIC_TAXONOMY_VERSION,
)


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
            "topic_taxonomy_version": TOPIC_TAXONOMY_VERSION,
        }


__all__ = ["GLiNER2TopicExtractor"]
