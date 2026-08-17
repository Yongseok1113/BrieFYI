"""임베딩 기반 그리디 클러스터링 (design doc 4절).

DB에 의존하지 않는 순수 로직만 담아서 테스트가 가볍다 — embed_fn을 주입받으므로
테스트에서는 실제 문장 임베딩 모델을 로드하지 않고 결정론적인 가짜 임베딩으로 검증한다.

알고리즘: 값을 등장 빈도 내림차순으로 정렬한 뒤, 이미 만들어진 클러스터의 대표(=canonical,
그 클러스터에서 가장 먼저 등장한 값이자 최빈값)와 코사인 유사도를 비교해 임계값 이상이면
합류시키고, 아니면 새 클러스터를 연다. 빈도 내림차순으로 처리하기 때문에 각 클러스터의
canonical은 자연스럽게 그 클러스터에서 가장 자주 쓰인 표현이 된다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Cluster:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    canonical_embedding: list[float] = field(default_factory=list)

    def alias_values(self) -> list[str]:
        """canonical 자기 자신도 aliases에 포함해 조회 시 검색이 쉽게 한다."""
        return [self.canonical] + self.aliases


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cluster_values(values_with_counts: list[tuple[str, int]], threshold: float,
                    embed_fn: EmbedFn) -> list[Cluster]:
    if not values_with_counts:
        return []

    ordered = sorted(values_with_counts, key=lambda vc: vc[1], reverse=True)
    values = [v for v, _count in ordered]
    embeddings = embed_fn(values)

    clusters: list[Cluster] = []
    for value, embedding in zip(values, embeddings):
        best_cluster, best_score = None, -1.0
        for cluster in clusters:
            score = _cosine_similarity(embedding, cluster.canonical_embedding)
            if score > best_score:
                best_cluster, best_score = cluster, score

        if best_cluster is not None and best_score >= threshold:
            best_cluster.aliases.append(value)
        else:
            clusters.append(Cluster(canonical=value, aliases=[], canonical_embedding=embedding))

    return clusters


def default_embed_fn(texts: list[str]) -> list[list[float]]:
    """실제 다국어 문장 임베딩 모델 (sentence-transformers, config.KEYWORD_EMBEDDING_MODEL 재사용)."""
    from sentence_transformers import SentenceTransformer

    from .config import config

    model = SentenceTransformer(config.KEYWORD_EMBEDDING_MODEL)
    return model.encode(texts).tolist()
