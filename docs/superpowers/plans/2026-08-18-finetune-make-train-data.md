# finetune/make_train_data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `raw_articles`(+`enrichment`)에서 기사를 읽어 클러스터링하고, 단발성 사실 기사를 선정해, Claude 채팅에 수동으로 붙여넣을 군집별 JSON 파일을 생성하는 `finetune/make_train_data/` 패키지를 만든다.

**Architecture:** 스펙 문서(`docs/superpowers/specs/2026-08-18-finetune-make-train-data-design.md`)를 그대로 구현한다 — DB 조회 → 엔티티 추출(enrichment 우선) → 4단계 클러스터링(시간창+엔티티 자카드 → 임베딩 정제 → 근접중복 병합) → 단발성 기사 선정 → 군집 파일 export. Claude API 자동 호출은 없다(파일 생성까지만 자동화).

**Tech Stack:** Python 3.11, `rag_latest.embed`(HF Inference API 재사용), 레포 루트 `db/db.py`(psycopg), `unittest`.

## Global Constraints

- `rag/`, `rag_experiment/`는 이 계획에서 참조는 하되 수정하지 않는다.
- 레포 루트 `db.py`/`config.py`를 재사용한다(sys.path 트릭, `finetune/src/summarize_ft/sources/*_export.py`와 동일 패턴) — `data_pipeline/`처럼 별도 이미지로 배포하지 않으므로 DB 코드를 복제하지 않는다.
- 각 태스크는 별도 커밋으로 남긴다. 커밋 메시지는 한국어로 작성한다.
- 새 파이썬 의존성을 추가하지 않는다(임베딩은 `rag_latest.embed` 재사용, 그 외는 표준 라이브러리).
- 클러스터링은 순수 함수 위주로 작성해 `embed_fn`을 주입받게 한다(`data_pipeline/clustering.py`와 동일 전략) — 테스트에서 실제 HF API를 호출하지 않는다.

---

### Task 1: `finetune/make_train_data/config.py`

**Files:**
- Create: `finetune/make_train_data/__init__.py` (빈 파일)
- Create: `finetune/make_train_data/config.py`
- Test: `finetune/make_train_data/tests/test_config.py`
- Create: `finetune/make_train_data/tests/__init__.py` (빈 파일)

**Interfaces:**
- Produces: `config.MTD_NARROW_WINDOW_HOURS: float`, `config.MTD_BROAD_WINDOW_DAYS: float`, `config.MTD_ENTITY_JACCARD_THRESHOLD: float`, `config.MTD_EMBED_SIM_THRESHOLD: float`, `config.MTD_DEDUP_THRESHOLD: float`, `config.MTD_MIN_CLUSTER_SIZE: int`, `config.MTD_ONEFACT_RATIO: float`, `config.MTD_MIN_ARTICLES: int` (모두 `Config` 클래스의 속성, 모듈 레벨 `config = Config()` 싱글턴)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# finetune/make_train_data/tests/test_config.py
from make_train_data.config import config


def test_기본값들이_로드된다():
    assert config.MTD_NARROW_WINDOW_HOURS == 72
    assert config.MTD_BROAD_WINDOW_DAYS == 28
    assert config.MTD_ENTITY_JACCARD_THRESHOLD == 0.3
    assert config.MTD_EMBED_SIM_THRESHOLD == 0.75
    assert config.MTD_DEDUP_THRESHOLD == 0.9
    assert config.MTD_MIN_CLUSTER_SIZE == 2
    assert config.MTD_ONEFACT_RATIO == 0.175
    assert config.MTD_MIN_ARTICLES == 20


def test_타입이_숫자다():
    assert isinstance(config.MTD_NARROW_WINDOW_HOURS, float)
    assert isinstance(config.MTD_MIN_CLUSTER_SIZE, int)
    assert isinstance(config.MTD_MIN_ARTICLES, int)
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
cd /home/ysoh1113/workspace/projects/BrieFYI
python -m pytest finetune/make_train_data/tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'make_train_data'` (아직 파일이 없음).

- [ ] **Step 3: `finetune/make_train_data/__init__.py`, `tests/__init__.py` 빈 파일 생성**

- [ ] **Step 4: `finetune/make_train_data/config.py` 작성**

```python
"""레포 루트 .env를 로드하는 설정. finetune/src/summarize_ft/sources/*_export.py와
동일한 sys.path 트릭을 쓴다 — data_pipeline/과 달리 별도 이미지로 배포하지 않으므로
DB 접속 정보(config.DATABASE_URL 등)는 레포 루트 config.py에서 그대로 가져다 쓴다.
여기 정의하는 값은 make_train_data 고유 설정(MTD_ 접두사)뿐이다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")


class Config:
    MTD_NARROW_WINDOW_HOURS = float(os.getenv("MTD_NARROW_WINDOW_HOURS", "72"))
    MTD_BROAD_WINDOW_DAYS = float(os.getenv("MTD_BROAD_WINDOW_DAYS", "28"))
    MTD_ENTITY_JACCARD_THRESHOLD = float(os.getenv("MTD_ENTITY_JACCARD_THRESHOLD", "0.3"))
    MTD_EMBED_SIM_THRESHOLD = float(os.getenv("MTD_EMBED_SIM_THRESHOLD", "0.75"))
    MTD_DEDUP_THRESHOLD = float(os.getenv("MTD_DEDUP_THRESHOLD", "0.9"))
    MTD_MIN_CLUSTER_SIZE = int(os.getenv("MTD_MIN_CLUSTER_SIZE", "2"))
    MTD_ONEFACT_RATIO = float(os.getenv("MTD_ONEFACT_RATIO", "0.175"))
    MTD_MIN_ARTICLES = int(os.getenv("MTD_MIN_ARTICLES", "20"))


config = Config()
```

- [ ] **Step 5: 통과 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_config.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add finetune/make_train_data/__init__.py finetune/make_train_data/config.py \
        finetune/make_train_data/tests/__init__.py finetune/make_train_data/tests/test_config.py
git commit -m "feat(make_train_data): 설정 모듈 추가"
```

---

### Task 2: `finetune/make_train_data/db.py`

**Files:**
- Create: `finetune/make_train_data/db.py`
- Test: `finetune/make_train_data/tests/test_db.py`

**Interfaces:**
- Consumes: 레포 루트 `db.db.get_conn()` (기존)
- Produces: `fetch_articles(since: str | None = None) -> list[dict]` — 각 dict는 `id, title, description, url, source, published_at, category, domain, entity, event, insights` 키를 가짐(enrichment 없으면 뒤 5개는 `None`)

- [ ] **Step 1: 실패하는 테스트 작성** (DB 통합 테스트, `tests/dbhelpers.py` 패턴 재사용)

```python
# finetune/make_train_data/tests/test_db.py
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unittest

from db.db import get_conn, init_db
from make_train_data.db import fetch_articles


class FetchArticlesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"DB 연결 불가: {exc}")

    def setUp(self):
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO raw_articles (digest_date, title, description, url, source, published_at)
                   VALUES (CURRENT_DATE, 'MTD 테스트 기사', '설명', 'https://test.invalid/mtd-1', 'test', now())
                   RETURNING id"""
            )
            self.article_id = cur.fetchone()["id"]

    def tearDown(self):
        with get_conn() as conn:
            conn.execute("DELETE FROM raw_articles WHERE url LIKE 'https://test.invalid/mtd-%%'")

    def test_enrichment_없어도_기사가_조회된다(self):
        rows = fetch_articles()
        matched = [r for r in rows if r["id"] == self.article_id]
        self.assertEqual(len(matched), 1)
        self.assertIsNone(matched[0]["entity"])

    def test_enrichment_있으면_함께_조회된다(self):
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO enrichment (raw_article_id, insights, category, domain, entity, event)
                   VALUES (%s, '[]', '기술', '["AI"]', '["NVIDIA"]', '["출시"]')""",
                (self.article_id,),
            )
        rows = fetch_articles()
        matched = [r for r in rows if r["id"] == self.article_id]
        self.assertEqual(matched[0]["category"], "기술")
        self.assertEqual(matched[0]["entity"], ["NVIDIA"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'make_train_data.db'`. (DB가 안 떠 있으면 대신 skip — 그 경우 Step 4까지 구현한 뒤 `docker compose up -d db`로 띄우고 재확인)

- [ ] **Step 3: `finetune/make_train_data/db.py` 작성**

```python
"""레포 루트 db.db.get_conn()을 재사용해 raw_articles(+enrichment)를 조회한다.
LEFT JOIN이다 — enrichment_export.py(INNER JOIN + pipeline_status='normalized'만)와
달리, 클러스터링은 enrichment 유무와 무관하게 모든 raw_articles를 대상으로 하고
enrichment 유무만 entity_extract.py의 분기 신호로 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def fetch_articles(since: str | None = None) -> list[dict[str, Any]]:
    from db.db import get_conn

    query = """
        SELECT a.id, a.title, a.description, a.url, a.source, a.published_at,
               e.category, e.domain, e.entity, e.event, e.insights
        FROM raw_articles a
        LEFT JOIN enrichment e ON e.raw_article_id = a.id
    """
    params: tuple = ()
    if since:
        query += " WHERE a.published_at >= %s"
        params = (since,)
    query += " ORDER BY a.published_at"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()
```

- [ ] **Step 4: 통과 확인** (DB 필요 — `docker compose up -d db`)

```bash
python -m pytest finetune/make_train_data/tests/test_db.py -v
```

Expected: 2 passed (또는 DB 접속 불가 시 skipped — 로컬에 DB가 없으면 이 상태로 다음 태스크 진행 가능, CI/실제 실행 전에 반드시 재확인).

- [ ] **Step 5: Commit**

```bash
git add finetune/make_train_data/db.py finetune/make_train_data/tests/test_db.py
git commit -m "feat(make_train_data): raw_articles+enrichment 조회 함수 추가"
```

---

### Task 3: `finetune/make_train_data/entity_extract.py`

**Files:**
- Create: `finetune/make_train_data/entity_extract.py`
- Test: `finetune/make_train_data/tests/test_entity_extract.py`

**Interfaces:**
- Produces: `extract(article: dict) -> tuple[list[str], str | None, list[str]]` (entities, category, domains)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# finetune/make_train_data/tests/test_entity_extract.py
from make_train_data.entity_extract import extract


def test_enrichment_있으면_그대로_반환():
    article = {"entity": ["NVIDIA", "TSMC"], "category": "기술", "domain": ["반도체"]}
    entities, category, domains = extract(article)
    assert entities == ["NVIDIA", "TSMC"]
    assert category == "기술"
    assert domains == ["반도체"]


def test_enrichment_없으면_시드_별칭_매칭():
    article = {"title": "엔비디아, TSMC와 공급 계약 확대", "description": "", "entity": None}
    entities, category, domains = extract(article)
    assert set(entities) == {"NVIDIA", "TSMC"}
    assert category is None
    assert domains == []


def test_영문_원문도_매칭된다():
    article = {"title": "OpenAI announces new partnership with Microsoft", "description": "", "entity": None}
    entities, _, _ = extract(article)
    assert set(entities) == {"OpenAI", "Microsoft"}


def test_매칭되는_엔티티_없으면_빈_리스트():
    article = {"title": "동네 카페, 신메뉴 출시", "description": "", "entity": None}
    entities, category, domains = extract(article)
    assert entities == []
    assert category is None
    assert domains == []


def test_description도_함께_검사한다():
    article = {"title": "업계 동향", "description": "삼성전자가 신규 라인을 발표했다", "entity": None}
    entities, _, _ = extract(article)
    assert entities == ["Samsung"]
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_entity_extract.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: `finetune/make_train_data/entity_extract.py` 작성**

```python
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
```

- [ ] **Step 4: 통과 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_entity_extract.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune/make_train_data/entity_extract.py finetune/make_train_data/tests/test_entity_extract.py
git commit -m "feat(make_train_data): 엔티티 추출(enrichment 우선 + 시드 별칭 fallback) 추가"
```

---

### Task 4: `finetune/make_train_data/embed.py` + `clustering.py`

**Files:**
- Create: `finetune/make_train_data/embed.py`
- Create: `finetune/make_train_data/clustering.py`
- Test: `finetune/make_train_data/tests/test_embed.py`
- Test: `finetune/make_train_data/tests/test_clustering.py`

**Interfaces:**
- Consumes: `entity_extract.extract`, `rag_latest.embed.embed_texts`
- Produces: `embed.embed_texts(texts: list[str]) -> list[list[float]]`, `clustering.Cluster`(dataclass: `cluster_id, window_type, articles, entities, event_type`), `clustering.cluster_articles(articles, *, narrow_window_hours, broad_window_days, entity_jaccard_threshold, embed_sim_threshold, dedup_threshold, min_cluster_size, entity_fn, embed_fn) -> tuple[list[Cluster], list[dict]]` (클러스터 목록, 어디에도 안 들어간 기사 목록)

- [ ] **Step 1: `embed.py` 실패 테스트 작성**

```python
# finetune/make_train_data/tests/test_embed.py
from unittest.mock import patch

from make_train_data.embed import embed_texts


def test_rag_latest_embed_texts를_그대로_호출한다():
    with patch("make_train_data.embed._embed_texts", return_value=[[0.1, 0.2]]) as mock_embed:
        result = embed_texts(["hello"])
    mock_embed.assert_called_once_with(["hello"])
    assert result == [[0.1, 0.2]]
```

- [ ] **Step 2: 실행해 실패 확인 → `embed.py` 작성**

```python
# finetune/make_train_data/embed.py
"""rag_latest.embed.embed_texts()를 그대로 호출하는 얇은 래퍼."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag_latest.embed import embed_texts as _embed_texts  # noqa: E402


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _embed_texts(texts)
```

```bash
python -m pytest finetune/make_train_data/tests/test_embed.py -v   # 1 passed
```

- [ ] **Step 3: `clustering.py` 실패 테스트 작성**

```python
# finetune/make_train_data/tests/test_clustering.py
from datetime import datetime, timedelta

from make_train_data.clustering import cluster_articles


def _article(id_, title, published_at, entity=None):
    return {"id": id_, "title": title, "description": "", "published_at": published_at, "entity": entity, "event": None}


def _entity_fn(article):
    return (article.get("entity") or [], None, [])


def _make_embed_fn(vectors: dict[str, list[float]]):
    """title -> 고정 벡터 매핑을 쓰는 가짜 embed_fn (data_pipeline 테스트 스타일)."""

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [vectors[t.strip()] for t in texts]

    return embed_fn


BASE_TIME = datetime(2026, 8, 1, 12, 0, 0)
DEFAULT_KWARGS = dict(
    narrow_window_hours=72,
    broad_window_days=28,
    entity_jaccard_threshold=0.3,
    embed_sim_threshold=0.75,
    dedup_threshold=0.9,
    min_cluster_size=2,
    entity_fn=_entity_fn,
)


def test_시간창_엔티티_모두_겹치면_한_클러스터로_묶인다():
    articles = [
        _article(1, "NVIDIA 공급 확대", BASE_TIME, ["NVIDIA"]),
        _article(2, "NVIDIA 공급 이슈 후속", BASE_TIME + timedelta(hours=10), ["NVIDIA"]),
    ]
    embed_fn = _make_embed_fn({"NVIDIA 공급 확대": [1.0, 0.0], "NVIDIA 공급 이슈 후속": [0.99, 0.01]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    narrow = [c for c in clusters if c.window_type == "narrow"]
    assert len(narrow) == 1
    assert {a["id"] for a in narrow[0].articles} == {1, 2}
    assert unclustered == []


def test_시간창을_벗어나면_안_묶인다():
    articles = [
        _article(1, "NVIDIA 공급 확대", BASE_TIME, ["NVIDIA"]),
        _article(2, "NVIDIA 공급 이슈 후속", BASE_TIME + timedelta(days=40), ["NVIDIA"]),
    ]
    embed_fn = _make_embed_fn({"NVIDIA 공급 확대": [1.0, 0.0], "NVIDIA 공급 이슈 후속": [0.99, 0.01]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    assert clusters == []
    assert len(unclustered) == 2


def test_엔티티가_다르면_안_묶인다():
    articles = [
        _article(1, "NVIDIA 공급 확대", BASE_TIME, ["NVIDIA"]),
        _article(2, "삼성 반도체 투자", BASE_TIME + timedelta(hours=1), ["Samsung"]),
    ]
    embed_fn = _make_embed_fn({"NVIDIA 공급 확대": [1.0, 0.0], "삼성 반도체 투자": [0.0, 1.0]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    assert clusters == []
    assert len(unclustered) == 2


def test_min_cluster_size_미만이면_unclustered로_간다():
    articles = [_article(1, "단독 기사", BASE_TIME, ["NVIDIA"])]
    embed_fn = _make_embed_fn({"단독 기사": [1.0, 0.0]})
    clusters, unclustered = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    assert clusters == []
    assert len(unclustered) == 1


def test_근접중복_클러스터는_병합된다():
    # 엔티티가 달라 자카드로는 안 묶이지만, 임베딩이 거의 같은 두 그룹은 근접중복으로 병합된다.
    articles = [
        _article(1, "A사 공급 확대", BASE_TIME, ["A사"]),
        _article(2, "A사 공급 후속", BASE_TIME + timedelta(hours=1), ["A사"]),
        _article(3, "동일 사건 다른 표현", BASE_TIME + timedelta(hours=2), ["B사"]),
        _article(4, "동일 사건 다른 표현 후속", BASE_TIME + timedelta(hours=3), ["B사"]),
    ]
    embed_fn = _make_embed_fn(
        {
            "A사 공급 확대": [1.0, 0.0],
            "A사 공급 후속": [0.99, 0.01],
            "동일 사건 다른 표현": [0.98, 0.02],
            "동일 사건 다른 표현 후속": [0.97, 0.03],
        }
    )
    clusters, _ = cluster_articles(articles, embed_fn=embed_fn, **DEFAULT_KWARGS)
    narrow = [c for c in clusters if c.window_type == "narrow"]
    assert len(narrow) == 1
    assert {a["id"] for a in narrow[0].articles} == {1, 2, 3, 4}
```

- [ ] **Step 4: 실행해 실패 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_clustering.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 5: `clustering.py` 작성**

```python
# finetune/make_train_data/clustering.py
"""기사 연관관계 클러스터링 (원 설계문서 3절 1~3단계 + 근접중복 병합).

시간창 필터 -> 엔티티 자카드 후보군(union-find) -> 임베딩 정제(그리디 threshold,
data_pipeline/clustering.py와 동일 전략) -> 근접중복 병합 순으로 실행한다.
원 설계문서 3절의 4단계("LLM 검증/병합", Claude 자동 호출)는 이 파이프라인에서
자동화하지 않는다 — 근접중복 병합으로 대체하고, 실제 검증은 cluster_export.py가
만드는 파일을 사람이 Claude 채팅에 전달하는 수동 단계에서 이뤄진다.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]
EntityFn = Callable[[dict], tuple[list[str], str | None, list[str]]]


@dataclass
class Cluster:
    cluster_id: str
    window_type: str  # "narrow" | "broad"
    articles: list[dict]
    entities: list[str]
    event_type: str | None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _article_text(article: dict) -> str:
    title = article.get("title") or ""
    description = article.get("description") or ""
    text = f"{title} {description}".strip()
    return text or title or "(제목 없음)"


def _candidate_groups(
    articles: list[dict],
    entity_sets: list[set[str]],
    window: timedelta,
    jaccard_threshold: float,
) -> list[list[int]]:
    n = len(articles)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        pub_i = articles[i].get("published_at")
        if pub_i is None:
            continue
        for j in range(i + 1, n):
            pub_j = articles[j].get("published_at")
            if pub_j is None or abs(pub_i - pub_j) > window:
                continue
            if _jaccard(entity_sets[i], entity_sets[j]) >= jaccard_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _refine_by_embedding(indices: list[int], embeddings: list[list[float]], threshold: float) -> list[list[int]]:
    if len(indices) <= 1:
        return [indices]

    sub_clusters: list[dict] = []
    for idx in indices:
        vec = embeddings[idx]
        best, best_score = None, -1.0
        for sc in sub_clusters:
            score = _cosine(vec, sc["centroid"])
            if score > best_score:
                best, best_score = sc, score
        if best is not None and best_score >= threshold:
            best["members"].append(idx)
        else:
            sub_clusters.append({"members": [idx], "centroid": vec})
    return [sc["members"] for sc in sub_clusters]


def _centroid(embeddings: list[list[float]], indices: list[int]) -> list[float]:
    vectors = [embeddings[i] for i in indices]
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]


def _groups_within_window(dated: list[dict], group_a: list[int], group_b: list[int], window: timedelta) -> bool:
    """두 그룹 중 어느 한 쌍이라도 시간창 안에 있으면 True. 근접중복 병합이 시간창을
    완전히 무시하고 임베딩만으로 멀리 떨어진 사건을 합치는 것을 막는다 — 이게 없으면
    1단계 시간창 필터가 4단계에서 무력화된다."""
    times_a = [dated[i]["published_at"] for i in group_a]
    times_b = [dated[i]["published_at"] for i in group_b]
    return any(abs(ta - tb) <= window for ta in times_a for tb in times_b)


def _merge_near_duplicates(
    groups: list[list[int]],
    embeddings: list[list[float]],
    dated: list[dict],
    window: timedelta,
    threshold: float,
) -> list[list[int]]:
    merged: list[dict] = []
    for group in groups:
        centroid = _centroid(embeddings, group)
        target = None
        for m in merged:
            if _groups_within_window(dated, group, m["members"], window) and _cosine(centroid, m["centroid"]) >= threshold:
                target = m
                break
        if target is not None:
            target["members"].extend(group)
            target["centroid"] = _centroid(embeddings, target["members"])
        else:
            merged.append({"members": list(group), "centroid": centroid})
    return [m["members"] for m in merged]


def cluster_articles(
    articles: list[dict],
    *,
    narrow_window_hours: float,
    broad_window_days: float,
    entity_jaccard_threshold: float,
    embed_sim_threshold: float,
    dedup_threshold: float,
    min_cluster_size: int,
    entity_fn: EntityFn,
    embed_fn: EmbedFn,
) -> tuple[list[Cluster], list[dict]]:
    """(클러스터 목록, 클러스터에 들지 못한 기사 목록)을 반환한다.

    narrow/broad 두 시간창은 서로 다른 목적(같은 사건 vs 같은 테마 흐름)이라 병행
    운영하며, 같은 기사가 양쪽에 각각 다른 클러스터로 들어갈 수 있다. "클러스터에
    들지 못했다"는 판정은 narrow 기준으로만 한다 — broad에만 걸리는 기사는 여전히
    개별 사건으로서는 고립돼 있어 단발성 후보로 취급하는 게 맞기 때문이다.
    """
    dated = [a for a in articles if a.get("published_at") is not None]
    if not dated:
        return [], list(articles)

    entity_sets = [set(entity_fn(a)[0]) for a in dated]
    texts = [_article_text(a) for a in dated]
    embeddings = embed_fn(texts)

    all_clusters: list[Cluster] = []
    clustered_indices: set[int] = set()

    for window_type, window in (
        ("narrow", timedelta(hours=narrow_window_hours)),
        ("broad", timedelta(days=broad_window_days)),
    ):
        candidate_groups = _candidate_groups(dated, entity_sets, window, entity_jaccard_threshold)
        refined: list[list[int]] = []
        for group in candidate_groups:
            refined.extend(_refine_by_embedding(group, embeddings, embed_sim_threshold))
        final_groups = _merge_near_duplicates(refined, embeddings, dated, window, dedup_threshold)

        for i, group in enumerate(final_groups):
            if len(group) < min_cluster_size:
                continue
            member_articles = [dated[idx] for idx in group]
            union_entities = sorted(set().union(*(entity_sets[idx] for idx in group)))
            raw_events = [dated[idx].get("event") for idx in group if dated[idx].get("event")]
            flat_events = [e for sub in raw_events for e in (sub if isinstance(sub, list) else [sub])]
            event_type = Counter(flat_events).most_common(1)[0][0] if flat_events else None

            all_clusters.append(
                Cluster(
                    cluster_id=f"c_{window_type}_{i}",
                    window_type=window_type,
                    articles=member_articles,
                    entities=union_entities,
                    event_type=event_type,
                )
            )
            if window_type == "narrow":
                clustered_indices.update(group)

    unclustered = [dated[i] for i in range(len(dated)) if i not in clustered_indices]
    return all_clusters, unclustered
```

- [ ] **Step 6: 통과 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_clustering.py finetune/make_train_data/tests/test_embed.py -v
```

Expected: 6 passed (5 clustering + 1 embed).

- [ ] **Step 7: Commit**

```bash
git add finetune/make_train_data/embed.py finetune/make_train_data/clustering.py \
        finetune/make_train_data/tests/test_embed.py finetune/make_train_data/tests/test_clustering.py
git commit -m "feat(make_train_data): 4단계 클러스터링(시간창+엔티티+임베딩+근접중복병합) 추가"
```

---

### Task 5: `finetune/make_train_data/onefact.py`

**Files:**
- Create: `finetune/make_train_data/onefact.py`
- Test: `finetune/make_train_data/tests/test_onefact.py`

**Interfaces:**
- Consumes: `clustering.Cluster`
- Produces: `select_onefact_candidates(clusters: list[Cluster], unclustered_articles: list[dict], *, target_ratio: float, total_pool_size: int) -> list[dict]`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# finetune/make_train_data/tests/test_onefact.py
from make_train_data.clustering import Cluster
from make_train_data.onefact import select_onefact_candidates


def _article(id_, event=None):
    return {"id": id_, "title": f"기사{id_}", "event": event}


def test_크기1_클러스터와_unclustered가_후보풀이_된다():
    clusters = [Cluster(cluster_id="c1", window_type="narrow", articles=[_article(1)], entities=[], event_type=None)]
    unclustered = [_article(2), _article(3)]
    selected = select_onefact_candidates(clusters, unclustered, target_ratio=1.0, total_pool_size=10)
    assert {a["id"] for a in selected} == {1, 2, 3}


def test_정기_발표성_event가_우선순위를_받는다():
    clusters = []
    unclustered = [
        _article(1, event=["실적발표"]),
        _article(2, event=["규제"]),
        _article(3, event=["제품출시"]),
    ]
    selected = select_onefact_candidates(clusters, unclustered, target_ratio=1.0, total_pool_size=2)
    selected_ids = {a["id"] for a in selected}
    assert selected_ids == {1, 3}  # 정기 발표성(실적발표/제품출시)이 "규제"보다 우선


def test_target_ratio로_선정_개수를_제한한다():
    unclustered = [_article(i) for i in range(10)]
    selected = select_onefact_candidates([], unclustered, target_ratio=0.2, total_pool_size=10)
    assert len(selected) == 2  # round(10 * 0.2)


def test_후보가_없으면_빈_리스트():
    selected = select_onefact_candidates([], [], target_ratio=0.175, total_pool_size=10)
    assert selected == []
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_onefact.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: `finetune/make_train_data/onefact.py` 작성**

```python
"""단발성 사실 기사 탐지·선정.

크기 1 클러스터 + 어디에도 클러스터링되지 못한 기사를 후보 풀로 삼고,
enrichment.event가 "정기 발표성"이면 우선순위를 높여 target_ratio만큼 선정한다.
enrichment.event는 data_pipeline이 LLM+synonym_table로 정규화한 값이라 고정
enum이 아니므로, ROUTINE_EVENT_TYPES는 코드 상수로 작게 시작해 synonym_table에
실제로 쌓인 canonical_value를 보고 조정하는 것을 전제로 한다. 최종 확정
(no_strong_insight 여부)은 이 함수가 하지 않는다 — 선정만 하고, 실제 라벨은
cluster_export.py가 만든 파일을 사람이 Claude 채팅에서 판단한다.
"""
from __future__ import annotations

ROUTINE_EVENT_TYPES = {"실적발표", "제품출시", "일반사실"}


def _is_routine(article: dict) -> bool:
    events = article.get("event") or []
    if isinstance(events, str):
        events = [events]
    return any(e in ROUTINE_EVENT_TYPES for e in events)


def select_onefact_candidates(
    clusters: list,
    unclustered_articles: list[dict],
    *,
    target_ratio: float,
    total_pool_size: int,
) -> list[dict]:
    pool: list[dict] = list(unclustered_articles)
    for cluster in clusters:
        if len(cluster.articles) == 1:
            pool.extend(cluster.articles)

    if not pool:
        return []

    routine = [a for a in pool if _is_routine(a)]
    rest = [a for a in pool if not _is_routine(a)]
    ordered = routine + rest  # 정기 발표성을 앞으로 정렬해 우선 선정되게 한다

    target_count = round(total_pool_size * target_ratio)
    target_count = max(0, min(target_count, len(ordered)))
    return ordered[:target_count]
```

- [ ] **Step 4: 통과 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_onefact.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune/make_train_data/onefact.py finetune/make_train_data/tests/test_onefact.py
git commit -m "feat(make_train_data): 단발성 사실 기사 탐지·선정 알고리즘 추가"
```

---

### Task 6: `finetune/make_train_data/cluster_export.py`

**Files:**
- Create: `finetune/make_train_data/cluster_export.py`
- Test: `finetune/make_train_data/tests/test_cluster_export.py`

**Interfaces:**
- Consumes: `clustering.Cluster`, `onefact.select_onefact_candidates`의 결과
- Produces: `export_clusters(clusters: list[Cluster], onefact_articles: list[dict], *, out_dir: Path, taxonomy_balance: bool = True) -> list[Path]`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# finetune/make_train_data/tests/test_cluster_export.py
import json
import tempfile
from pathlib import Path

from make_train_data.clustering import Cluster
from make_train_data.cluster_export import export_clusters


def _article(id_, title="제목", event_type=None):
    return {
        "id": id_, "title": title, "description": "설명", "url": f"https://x/{id_}",
        "source": "test", "published_at": None, "insights": None, "category": None, "domain": None,
    }


def test_클러스터당_파일_하나가_생성된다():
    clusters = [
        Cluster(cluster_id="c1", window_type="narrow", articles=[_article(1), _article(2)], entities=["NVIDIA"], event_type="공급망"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_clusters(clusters, [], out_dir=Path(tmp))
        assert len(paths) == 1
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        assert data["cluster_id"] == "c1"
        assert data["window_type"] == "narrow"
        assert len(data["articles"]) == 2
        assert "claude_prompt" in data
        assert data["no_strong_insight_hint"] is False


def test_단발성_기사는_no_strong_insight_hint_true로_별도_파일에_들어간다():
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_clusters([], [_article(1), _article(2)], out_dir=Path(tmp))
        assert len(paths) == 1
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        assert data["no_strong_insight_hint"] is True
        assert len(data["articles"]) == 2


def test_taxonomy_balance가_켜지면_한쪽으로_쏠린_event_type을_일부_제외한다():
    clusters = [
        Cluster(cluster_id=f"c{i}", window_type="narrow", articles=[_article(i), _article(i + 100)], entities=[], event_type="M&A")
        for i in range(10)
    ] + [
        Cluster(cluster_id="c_rare", window_type="narrow", articles=[_article(999), _article(998)], entities=[], event_type="규제")
    ]
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_clusters(clusters, [], out_dir=Path(tmp), taxonomy_balance=True)
        event_types = [json.loads(p.read_text(encoding="utf-8"))["event_type"] for p in paths]
        assert "규제" in event_types  # 희소 유형은 반드시 포함
        assert event_types.count("M&A") < 10  # 압도적 다수인 유형은 일부 제외됨
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_cluster_export.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: `finetune/make_train_data/cluster_export.py` 작성**

```python
"""클러스터(+단발성 기사)를 Claude 채팅에 붙여넣을 JSON 파일로 저장한다.

taxonomy_balance=True면 event_type 분포를 보고 특정 유형이 압도적으로 많을 때
초과분을 샘플링에서 제외한다(원 설계문서 5절 골드셋 규칙을 학습셋 생성에도 적용) —
가장 흔한 유형이라도 최소 개수는 남기고, 나머지 유형 개수의 평균을 넘는 만큼만 잘라낸다.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .clustering import Cluster

CLAUDE_PROMPT_TEMPLATE = (
    "다음 기사 묶음이 정말 같은 사건/테마를 다루는지 먼저 확인해줘. 다루지 않는다면 "
    "어떻게 나눠야 하는지 알려줘. 같은 사건/테마가 맞다면, facts/insights"
    "(전략적_의도|모순_긴장|선행지표|이해관계자_영향|시장_신호|리스크_기회 중 최소 2종 이상 "
    "시도, 억지로 6종 다 채우지 않음)/no_strong_insight를 판단해서 아래 스키마로 답해줘.\n\n"
    '{"facts": ["..."], "insights": [{"claim": "...", "type": "...", "evidence": ["..."], '
    '"confidence": "high|medium|low", "reasoning": "..."}], "no_strong_insight": false}\n\n'
    "추가로 이 클러스터에 대해 종합적 판단을 요구하는 테스트용 질문과 모범 답안도 하나 만들어줘."
)


def _article_payload(article: dict) -> dict:
    payload = {
        "title": article.get("title"),
        "description": article.get("description"),
        "source": article.get("source"),
        "published_at": str(article.get("published_at")) if article.get("published_at") else None,
        "url": article.get("url"),
    }
    if article.get("insights") or article.get("category") or article.get("domain"):
        payload["enrichment_hint"] = {
            "insights": article.get("insights"),
            "category": article.get("category"),
            "domain": article.get("domain"),
        }
    return payload


def _cluster_to_dict(cluster: Cluster, *, no_strong_insight_hint: bool) -> dict:
    return {
        "cluster_id": cluster.cluster_id,
        "window_type": cluster.window_type,
        "no_strong_insight_hint": no_strong_insight_hint,
        "entities": cluster.entities,
        "event_type": cluster.event_type,
        "articles": [_article_payload(a) for a in cluster.articles],
        "claude_prompt": CLAUDE_PROMPT_TEMPLATE,
    }


def _apply_taxonomy_balance(clusters: list[Cluster]) -> list[Cluster]:
    counts = Counter(c.event_type for c in clusters if c.event_type)
    if len(counts) <= 1:
        return clusters

    other_counts = sorted(v for k, v in counts.items())[:-1]
    cap = max(1, round(sum(other_counts) / len(other_counts))) if other_counts else max(counts.values())

    kept: list[Cluster] = []
    seen_count: dict[str, int] = {}
    for cluster in clusters:
        key = cluster.event_type
        if key is None:
            kept.append(cluster)
            continue
        seen_count[key] = seen_count.get(key, 0) + 1
        if seen_count[key] <= max(cap, 1):
            kept.append(cluster)
    return kept


def export_clusters(
    clusters: list[Cluster],
    onefact_articles: list[dict],
    *,
    out_dir: Path,
    taxonomy_balance: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clusters = _apply_taxonomy_balance(clusters) if taxonomy_balance else clusters

    paths: list[Path] = []
    for cluster in target_clusters:
        path = out_dir / f"{cluster.cluster_id}.json"
        path.write_text(
            json.dumps(_cluster_to_dict(cluster, no_strong_insight_hint=False), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(path)

    if onefact_articles:
        onefact_cluster = Cluster(
            cluster_id="c_onefact_batch",
            window_type="narrow",
            articles=onefact_articles,
            entities=[],
            event_type=None,
        )
        path = out_dir / "c_onefact_batch.json"
        path.write_text(
            json.dumps(_cluster_to_dict(onefact_cluster, no_strong_insight_hint=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(path)

    return paths
```

- [ ] **Step 4: 통과 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_cluster_export.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune/make_train_data/cluster_export.py finetune/make_train_data/tests/test_cluster_export.py
git commit -m "feat(make_train_data): 군집 파일 export(taxonomy 균형 샘플링 포함) 추가"
```

---

### Task 7: `finetune/make_train_data/cli.py`

**Files:**
- Create: `finetune/make_train_data/cli.py`
- Test: `finetune/make_train_data/tests/test_cli.py`

**Interfaces:**
- Consumes: 모든 이전 태스크의 함수
- Produces: `run(out_dir: Path, since: str | None = None) -> int` (0=성공, 1=데이터 부족), CLI 진입점 `python -m make_train_data.cli run --out-dir ... [--since ...]`

- [ ] **Step 1: 실패하는 테스트 작성** (DB/임베딩 없이 `run()`의 오케스트레이션 로직만 mock으로 검증)

```python
# finetune/make_train_data/tests/test_cli.py
import tempfile
from pathlib import Path
from unittest.mock import patch

from make_train_data.cli import run


def test_기사가_MIN_ARTICLES_미만이면_1을_반환하고_export하지_않는다():
    with patch("make_train_data.cli.fetch_articles", return_value=[{"id": 1}] * 5), \
         patch("make_train_data.cli.config") as mock_config, \
         patch("make_train_data.cli.export_clusters") as mock_export:
        mock_config.MTD_MIN_ARTICLES = 20
        with tempfile.TemporaryDirectory() as tmp:
            code = run(out_dir=Path(tmp))
    assert code == 1
    mock_export.assert_not_called()


def test_기사가_충분하면_전체_파이프라인을_거쳐_export한다():
    articles = [{"id": i} for i in range(25)]
    with patch("make_train_data.cli.fetch_articles", return_value=articles), \
         patch("make_train_data.cli.config") as mock_config, \
         patch("make_train_data.cli.cluster_articles", return_value=([], articles)) as mock_cluster, \
         patch("make_train_data.cli.select_onefact_candidates", return_value=articles[:4]) as mock_onefact, \
         patch("make_train_data.cli.export_clusters", return_value=[Path("x.json")]) as mock_export:
        mock_config.MTD_MIN_ARTICLES = 20
        mock_config.MTD_NARROW_WINDOW_HOURS = 72
        mock_config.MTD_BROAD_WINDOW_DAYS = 28
        mock_config.MTD_ENTITY_JACCARD_THRESHOLD = 0.3
        mock_config.MTD_EMBED_SIM_THRESHOLD = 0.75
        mock_config.MTD_DEDUP_THRESHOLD = 0.9
        mock_config.MTD_MIN_CLUSTER_SIZE = 2
        mock_config.MTD_ONEFACT_RATIO = 0.175
        with tempfile.TemporaryDirectory() as tmp:
            code = run(out_dir=Path(tmp))
    assert code == 0
    mock_cluster.assert_called_once()
    mock_onefact.assert_called_once()
    mock_export.assert_called_once()
```

- [ ] **Step 2: 실행해 실패 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: `finetune/make_train_data/cli.py` 작성**

```python
"""make_train_data 파이프라인 진입점.

    python -m make_train_data.cli run --out-dir finetune/make_train_data/output --since 2026-08-01
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .cluster_export import export_clusters
from .clustering import cluster_articles
from .config import config
from .db import fetch_articles
from .embed import embed_texts
from .entity_extract import extract as entity_extract
from .onefact import select_onefact_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(out_dir: Path, since: str | None = None) -> int:
    articles = fetch_articles(since=since)
    if len(articles) < config.MTD_MIN_ARTICLES:
        logger.info(
            "현재 raw_articles가 %d건뿐입니다(최소 %d건 필요). data_pipeline을 더 돌려 "
            "데이터를 쌓은 뒤 다시 실행하세요.",
            len(articles), config.MTD_MIN_ARTICLES,
        )
        return 1

    clusters, unclustered = cluster_articles(
        articles,
        narrow_window_hours=config.MTD_NARROW_WINDOW_HOURS,
        broad_window_days=config.MTD_BROAD_WINDOW_DAYS,
        entity_jaccard_threshold=config.MTD_ENTITY_JACCARD_THRESHOLD,
        embed_sim_threshold=config.MTD_EMBED_SIM_THRESHOLD,
        dedup_threshold=config.MTD_DEDUP_THRESHOLD,
        min_cluster_size=config.MTD_MIN_CLUSTER_SIZE,
        entity_fn=entity_extract,
        embed_fn=embed_texts,
    )
    onefact_articles = select_onefact_candidates(
        clusters, unclustered, target_ratio=config.MTD_ONEFACT_RATIO, total_pool_size=len(articles)
    )
    paths = export_clusters(clusters, onefact_articles, out_dir=out_dir)
    logger.info("클러스터 %d개 + 단발성 배치 파일 %d개 -> %s", len(clusters), len(paths) - len(clusters), out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="군집 기반 인사이트 학습 데이터 생성")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="DB에서 기사를 읽어 군집 파일을 생성한다")
    run_parser.add_argument("--out-dir", type=Path, default=Path("finetune/make_train_data/output"))
    run_parser.add_argument("--since", default=None, help="이 날짜(YYYY-MM-DD) 이후 기사만 대상")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run(out_dir=args.out_dir, since=args.since)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 통과 확인**

```bash
python -m pytest finetune/make_train_data/tests/test_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 5: 전체 make_train_data 테스트 회귀 확인**

```bash
python -m pytest finetune/make_train_data/tests -v
```

Expected: 전부 통과 (test_db.py는 DB 연결 가능 여부에 따라 skip 가능).

- [ ] **Step 6: Commit**

```bash
git add finetune/make_train_data/cli.py finetune/make_train_data/tests/test_cli.py
git commit -m "feat(make_train_data): CLI 진입점 추가 (python -m make_train_data.cli run)"
```

---

### Task 8: 노트북 (`finetune/notebooks/make_train_data_colab.ipynb`)

**Files:**
- Create: `finetune/notebooks/make_train_data_colab.ipynb`

**배경:** 기존 `qlora_qwen3_8b_colab.ipynb`(학습 전용)와 독립된 새 노트북. `make_train_data`는 GPU가 필수는 아니지만(임베딩은 HF Inference API), 로컬 환경 준비가 안 됐을 때 Colab에서 바로 돌릴 수 있게 한다.

- [ ] **Step 1: 노트북 셀 구성** (기존 `qlora_qwen3_8b_colab.ipynb`의 "번들 업로드" 셀 패턴을 참고해 동일한 스타일로 작성)

셀 순서:
1. 마크다운: 제목 + 개요 ("make_train_data 파이프라인을 Colab에서 실행 — DB 접속 정보만 있으면 로컬 환경 없이 군집 파일을 생성할 수 있다")
2. 번들 업로드: `finetune/make_train_data/`, `finetune/src/summarize_ft/` 없이 `make_train_data/` 폴더 + 레포 루트 `db/`, `config.py`, `rag_latest/`, `requirements.txt`(임베딩·DB에 필요한 최소 의존성)만 zip으로 묶어 업로드하는 셀 (`files.upload()` + `zipfile`)
3. `.env` 구성 셀: `DATABASE_URL`, `HF_TOKEN`을 Colab Secrets 또는 직접 입력으로 받아 파일로 씀
4. 의존성 설치: `!pip install -q psycopg[binary] python-dotenv requests`
5. 실행: `!python -m make_train_data.cli run --out-dir output --since 2026-08-01`
6. 결과 다운로드: `output/` 디렉터리를 zip으로 묶어 `files.download()`

```python
# Step 2 셀 예시 (번들 업로드)
from google.colab import files
import zipfile, os

print("make_train_data_bundle.zip을 선택하세요 (make_train_data/, db/, config.py, rag_latest/ 포함)")
uploaded = files.upload()
bundle_name = list(uploaded.keys())[0]
with zipfile.ZipFile(bundle_name, "r") as z:
    z.extractall("/content")
os.chdir("/content")
```

```python
# Step 3 셀 예시 (.env 구성)
import os
from getpass import getpass

database_url = os.environ.get("DATABASE_URL") or getpass("DATABASE_URL: ")
hf_token = os.environ.get("HF_TOKEN") or getpass("HF_TOKEN: ")
with open(".env", "w") as f:
    f.write(f"DATABASE_URL={database_url}\nHF_TOKEN={hf_token}\n")
```

- [ ] **Step 2: 노트북 파일 생성** — 위 셀 구성을 `nbformat` 없이 직접 JSON 구조로 작성(`qlora_qwen3_8b_colab.ipynb`와 같은 `nbformat: 4` 구조를 그대로 따름). 마크다운 셀은 `cell_type: "markdown"`, 코드 셀은 `cell_type: "code"`, `outputs: []`, `execution_count: null`.

- [ ] **Step 3: 노트북 유효성 확인**

```bash
python -c "import json; json.load(open('finetune/notebooks/make_train_data_colab.ipynb'))" && echo "유효한 JSON"
```

Expected: `유효한 JSON` 출력, 예외 없음.

- [ ] **Step 4: Commit**

```bash
git add finetune/notebooks/make_train_data_colab.ipynb
git commit -m "feat(make_train_data): Colab 실행용 노트북 추가"
```

---

## 실행 후 전체 회귀 확인 (모든 태스크 완료 후)

```bash
python -m pytest finetune/make_train_data/tests -v
python -c "import json; json.load(open('finetune/notebooks/make_train_data_colab.ipynb'))"
```

DB가 떠 있는 상태(`docker compose up -d db`)에서 `test_db.py`까지 포함해 전부 통과해야 한다.
