# BrieFYI RAG

현재 PostgreSQL의 `raw_articles` URL에서 기사 본문을 best-effort로 읽어 BGE-M3
tokenizer로 청킹하고, GLiNER2 4-Layer metadata와 embedding을 함께 저장한 뒤 vector,
text 또는 hybrid 방식으로 검색한다.

현재 digest 파이프라인이나 `graph/pipeline.py`에는 연결하지 않았다. 검색 기능과
DB 계약을 먼저 검증한 뒤, 향후 뉴스 agent가 호출하는 Tool로 `retrieve()`를 감싸는
방향을 전제로 한다.

## 현재 구현 범위

| 파일 | 역할 |
| --- | --- |
| `chunk.py` | 본문 우선 텍스트 구성과 BGE-M3 500/50 token chunk 생성 |
| `embed.py` | Hugging Face Inference API로 BGE-M3 dense embedding 생성 |
| `indexer.py` | 본문·GLiNER2 topic/Event·chunk embedding을 함께 저장하는 bounded worker |
| `topic_taxonomy.py` | 실제 프로젝트용 Category/Domain 고정 taxonomy |
| `topic_extractor.py` | GLiNER2 Category/Domain/Entity 추출과 article-level 집계 |
| `event_taxonomy.py` | AI·기술 relation 10개와 role mapping, 추출 version |
| `event_extractor.py` | GLiNER2 relation 추출과 false-positive 후처리 |
| `event_indexer.py` | 기존 article chunk의 구조화 Event/Argument 저장 |
| `retriever.py` | vector, text, normalized hybrid, RRF hybrid 검색 |
| `rag_function_test.ipynb` | DB 확인부터 인덱싱·검색까지 단계별 기능 테스트 |

```text
raw_articles(title, description)
    -> 기사 URL 본문 수집 (실패 시 title+description)
    -> build_article_text()
    -> BGE-M3 tokenizer split_text(500, overlap=50)
    -> article_chunks
    -> embed_texts(BAAI/bge-m3)
    -> chunk_embeddings
    -> article_topics(Category/Domain/Entity)
    -> article_events(Event/Arguments)

사용자 query
    -> vector search + text search
    -> weighted RRF
    -> Category/Domain 일치 가산점
    -> 관련 chunk 목록
```

이 모듈의 인덱싱 입력은 현재 BrieFYI DB의 `raw_articles`로 제한한다.

## 현재 완성 범위와 독립 실행성

RAG 핵심인 `raw_articles -> article_chunks -> chunk_embeddings -> retrieve()`는 함수
단위로 완성되어 있고 `graph/pipeline.py`를 import하지 않고 독립 실행할 수 있다. 다만
`rag` 전체를 하나의 완성된 자동 파이프라인이나 서비스로 실행하는 상태는 아니다.

| 기능 | 공개 진입점 | 현재 상태 |
| --- | --- | --- |
| 수집+4-Layer+embedding | `python -m rag.ingestion` | GNews 신규 기사 end-to-end 처리 |
| chunk/4-Layer/embedding | `index_article(s)`, `index_all_articles()` / `python -m rag.indexer` | 독립 실행 가능 |
| vector/text/hybrid 검색 | `retrieve()` / `python -m rag.retriever` | 독립 실행 가능 |
| Category/Domain/Entity prototype | `python -m rag.topic_indexer` | 명시 ID 기반 독립 실행 가능 |
| 구조화 Event 저장 | `index_event_articles()` / `python -m rag.event_indexer` | 명시 ID 기반 local 독립 실행 가능 |
| 전체 RAG package | `python -m rag` | `rag/__main__.py`가 없어 지원하지 않음 |

따라서 향후 Agent Tool은 기존 함수를 얇게 감싸면 된다. `python -m rag.ingestion`은
GNews 수집부터 GLiNER2 4-Layer와 RAG 저장까지 수행하지만, 검색 답변 생성과 digest 발송을
포함하는 `python -m rag` 전체 application은 아직 없다.

## 현재 pipeline에 최소 기능 통합 가능한가

**코드 계약 기준으로는 가능하다.** 현재 `store_raw_node`가 article ID가 아니라
`inserted_count`만 반환하지만, 이를 변경하지 않고 다음 adapter node를 추가할 수 있다.

```text
별도 ingestion worker
  -> fetch_news()
  -> insert_articles()
  -> 이번 신규 URL의 article ID 조회
  -> index_articles(article_ids)
       - 본문/chunk/embedding
       - GLiNER2 Category/Domain/Entity
       - GLiNER2 Event/Arguments
```

`index_all_articles()`는 DB에서 embedding이 없는 기사를 직접 선택하고, 처리 결과에
`article_id`를 반환한다. 따라서 최소 통합을 위해 `insert_articles() -> int` 또는 현재
`PipelineState.inserted_count` 계약을 바꿀 필요는 없다.

다만 실제 shared pipeline에 넣기 전에는 다음 세 가지를 보완해야 한다.

1. HF embedding이나 GLiNER2 indexing 실패가 요약·메일 발송을 중단하지 않도록 adapter가
   예외를 잡고 별도 `rag_error`/결과를 남기는 best-effort 경계가 필요하다.
2. 현재 GitHub Actions는 pgvector가 없는 `postgres:16-alpine`을 사용하고 `HF_TOKEN`도
   전달하지 않는다. `main.py`의 `init_db()`가 이미 vector schema를 적용하므로, RAG node를
   추가하기 전에도 현재 RAG_1 workflow는 vector extension 단계에서 실패할 가능성이 있다.
3. 현재 digest image의 공유 requirements에는 GLiNER2/torch가 없다. 4-Layer worker는 우선
   local 또는 별도 worker로 두고, inline node로 넣지 않는 편이 현실적이다.

따라서 최소 통합 가능성은 다음처럼 구분한다.

- **RAG ingestion worker:** 별도 CLI로 동작하며 main graph adapter와 실패 격리는 아직 필요.
- **검색 Tool:** `retrieve(query, top_k)`를 그대로 감싸 등록 가능. 다만 현재 Agent들은
  주입된 함수를 고정 순서로 호출할 뿐, LLM이 검색 필요 여부를 판단하는 tool loop는 없다.
- **구조화 Event:** local indexer에는 연결됐지만,
  현재 Docker/GHA pipeline에는 별도 worker/runtime 결정 전까지 연결하지 않는다.
- **완전한 production RAG pipeline:** 아직 아님. 자동 후보 제한, 관찰/재시도, Agent
  tool loop와 배포 환경 정리가 남아 있다.

현재 변경되지 않은 `graph.pipeline.build_graph()`는 local 환경에서
`CompiledStateGraph`로 정상 compile되는 것을 확인했다. 위 optional node를 실제로 추가한
graph는 아직 구현하거나 실행하지 않았다.

## 설정과 실행 조건

프로젝트 루트의 `.env`에는 embedding과 prototype Entity 추출에 사용할 API 키가 필요하다.

```dotenv
HF_TOKEN="your_huggingface_token"
HF_EMBEDDING_MODEL="BAAI/bge-m3"
ANTHROPIC_API_KEY="your_anthropic_api_key"
```

고정 가능한 값은 환경변수로 늘리지 않고 소스 설정을 사용한다.

- embedding dimension: `1024`
- HF 요청 timeout: `120`초
- chunk size: `500` token
- overlap: `50` token

필요한 Python 의존성은 `anthropic`, `requests`, `psycopg`, `pgvector`, `python-dotenv`이며
프로젝트 `requirements.txt`에 포함되어 있다. DB는 pgvector 확장이 설치된
PostgreSQL 16이어야 한다.

본문/4-Layer worker는 추가로 `beautifulsoup4`, `transformers`, GLiNER2가 필요하다.
이번 구현에서는 공유 requirements를 변경하지 않았으므로 해당 package가 이미 설치된 local
worker 환경에서 실행한다. 누락 시 lazy import 지점에서 필요한 package를 명시해 실패한다.

구조화 Event는 local worker용 별도 의존성을 사용하며 공유 `requirements.txt`에는
포함하지 않는다.

```bash
pip install -r rag/requirements-event.txt
```

```bash
docker compose up -d db
```

호스트 Python은 기본적으로 `localhost:5432`, Compose의 `digest` 컨테이너는
`db:5432`로 접속한다. `.env`를 수정한 뒤 이미 import된 `config` 값이 남아 있다면
노트북 kernel을 재시작해야 한다.

## DB 구조

`db/vector_schema.sql`이 다음 관계를 추가한다.

```text
raw_articles 1 --- N article_chunks 1 --- N chunk_embeddings
     |                  |
     |                  +--- N article_events 1 --- N article_event_arguments
     |
     +--- 1 article_event_index_status
```

`article_chunks`는 `(article_id, chunk_index)`, `chunk_embeddings`는
`(chunk_id, embedding_model)`을 unique key로 사용한다. embedding 행에는 모델 이름,
선언된 차원, 실제 vector를 함께 저장하며
`vector_dims(embedding) = embedding_dimension` check로 차원을 검증한다.

동일 chunk에 여러 모델의 벡터를 함께 둘 수 있고, 검색할 때는 query와 같은
`embedding_model`, `embedding_dimension`만 선택한다.

## GLiNER2 4-Layer indexing

Category/Domain/Entity는 `article_topics`, Event는 구조화 Event 테이블에 저장한다.
4-Layer 자체를 별도 vector로 만들지는 않는다.

```text
기사 제목 + URL 본문
  -> GLiNER2 business_tech_v1
       Category: 경제/기술/금융/산업/기타 중 하나
       Domain: 반도체/AI/배터리/.../게임/기타 중 복수
       Entity: 핵심 회사/기관/인물/제품 최대 3개
  -> article_topics UPSERT

raw_articles 1 --- 1 article_topics
     |
     +--- N article_chunks 1 --- N chunk_embeddings
```

- `category`: 기술, 경제 같은 대분류
- `domains`: AI, 반도체 같은 세부 도메인
- `entities`: 기사에서 핵심적으로 다루는 회사·기관·인물·제품, 최대 3개

기존 `article_topics.events` free-text prototype은 더 이상 생성하거나 갱신하지 않는다.
기존 DB에 남아 있는 컬럼과 데이터는 자동 삭제하지 않는다.

`topic_text`와 topic `embedding`은 후속 실험용 컬럼이며 현재는 NULL로 둔다. 기존
기사의 chunk embedding은 다시 만들지 않고 그대로 연결해서 사용한다.

GNews 신규 기사를 수집하면서 4-Layer와 RAG를 함께 저장하려면 다음을 실행한다.

```bash
python -m rag.ingestion --keyword AI --days 1 --max-results 10 --device cuda

# 기존 article ID를 본문부터 다시 처리
python -m rag.indexer --article-ids 19 20 21 --device cuda
```

GPU가 없는 환경에서는 `--device cpu`를 명시한다. 기존 수동 Category/Domain CLI인
`python -m rag.topic_indexer`도 호환용으로 유지하지만 기본 ingestion 경로는 GLiNER2가
Category/Domain/Entity를 자동 생성한다.

## 구조화 Event indexing

Event는 자유 명사구가 아니라 고정 relation과 role별 argument로 저장한다.

```text
invested_in
  investor: NVIDIA
  investee: OpenAI
```

1차 taxonomy는 `ai_tech_v1`의 10개 relation이며, 추출 설정과 후처리 version은
`gliner2_event_v1`이다. GLiNER2가 같은 argument 쌍에 여러 relation을 반환하면 confidence가
가장 높은 하나만 남긴다. 양쪽 argument가 같은 window에서 추출된 Entity와 일치하지 않거나,
정규화 후 head와 tail이 같으면 저장하지 않는다. `partnered_with`만 대칭 relation으로
처리한다.

GLiNER2 1.3.2 `multi-v1`에서는 Entity와 Relation을 한 schema에 함께 넣었을 때 relation은
나오지만 Entity가 빈 결과가 되는 사례가 확인됐다. 따라서 같은 window에 Entity schema와
Relation schema를 각각 실행하고, 두 결과를 합쳐 argument-Entity 일치 여부를 검사한다.

기본 indexer는 topic metadata 저장 후 같은 GLiNER2 model을 재사용해 Event도 처리한다.
Event만 명시적으로 재처리할 때는 다음 CLI를 사용한다.

```bash
python -m rag.event_indexer \
  --article-ids 19 20 21 \
  --device cuda
```

`--device cuda`에서 CUDA를 사용할 수 없으면 CPU로 자동 전환하지 않고 실패한다. CPU를
의도했다면 `--device cpu`를 사용한다. 같은 chunk와 model/taxonomy/extraction version으로
완료한 기사는 건너뛰며, 다시 처리하려면 `--force`를 추가한다. Event가 없는 기사도
`completed`, `event_count=0`으로 기록한다.

Event indexer는 `raw_articles.pipeline_status`나 별도 `develop/data_pipeline`의
`enrichment.event`에 의존하지 않는다. Event embedding, Event 검색, Entity FK 연결,
pipeline/Agent Tool은 후속 범위다.

### 현재 model smoke test 한계

캐시된 `fastino/gliner2-multi-v1`을 CPU에서 6개 짧은 예문으로 확인했다. 투자, 인수,
협력은 각각 `invested_in`, `acquired`, `partnered_with`로 남았고, 날씨와 일반 시장 발언
예문은 Event가 없었다. 다만 `OpenAI released the GPT-X model.`은 `released`가 아니라
`developed`로 선택됐고, 협력 예문에는 추가 `developed` 후보도 남았다. 따라서 현재 결과는
구조와 실행 계약을 확인한 baseline이며, relation별 의미 정확도가 검증됐다는 뜻은 아니다.

## 청킹과 Embedding

`raw_articles`에는 본문을 추가 저장하지 않는다. indexer가 URL에서 본문을 읽어 성공하면
`title + body`, 실패하면 `title + description`을 BGE-M3 fast tokenizer로 나눈다.

```python
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
```

`embed_texts()`는 문자열 N개를 HF Inference API에 전달하고 정규화된
`[N, 1024]` 벡터를 반환한다. 응답의 벡터 개수, 차원, NaN/Infinity, L2 norm을
검사하며 `embed_query()`도 같은 모델과 정규화 방식을 사용한다.

## 인덱싱

```python
from rag.indexer import index_all_articles, index_article, index_articles

all_results = index_all_articles()
one_result = index_article(19)
selected_results = index_articles([19, 20, 21])
```

`index_all_articles()`는 `chunk_embeddings`가 아직 없는 기사만 선택한다. 재실행하면
이미 완료된 기사는 HF API에 다시 보내지 않는다.

`index_article()`과 `index_articles()`를 명시적으로 호출하면 기존 기사도 다시 처리한다.
저장 시 DB unique key를 기준으로 upsert하므로 같은 모델의 행이 계속 늘어나지 않는다.
원본 chunk 텍스트가 달라지면 기존 chunk에 연결된 embedding을 삭제한 뒤 현재 텍스트로
다시 저장한다.

## 검색 방식

- `vector`: query와 chunk embedding의 cosine, L2 또는 inner product 검색
- `text`: PostgreSQL `ts_rank_cd` 기반 full-text 검색
- `hybrid`: vector와 text 후보 순위를 weighted RRF로 결합

```python
from rag.retriever import retrieve

vector_rows = retrieve("Claude", top_k=5, search_mode="vector", metric="cosine")
text_rows = retrieve("Claude", top_k=5, search_mode="text")
```

text 검색은 현재 `websearch_to_tsquery('simple', ...)`를 사용한다. `chunk_text`에
제목이 이미 포함되므로 검색 SQL에서 제목을 다시 이어 붙이지 않는다.

기본 hybrid 식은 다음과 같다.

```text
score = vector_weight / (rrf_k + vector_rank)
      + text_weight   / (rrf_k + text_rank)
```

한 검색기에 없는 후보의 해당 기여도는 0이다. 같은 원점수에는 같은 dense rank를
부여한다. 결과에는 원점수와 함께 `vector_rank`, `text_rank`,
`vector_rrf_score`, `text_rrf_score`를 남긴다.

```text
text_score: 0.3, 0.3, 0.2
text_rank:    1,   1,   2
```

```python
from rag.retriever import retrieve

rows = retrieve(
    query="Claude",
    top_k=5,
    search_mode="hybrid",
    fusion_method="rrf",
    candidate_k=50,
    rrf_k=60,
    vector_weight=0.7,
    text_weight=0.3,
    category="기술",
    domains=["AI"],
)
```

Category/Domain은 hard filter가 아니다. 먼저 기존 검색 후보를 만든 다음 일치 항목에
score scale 기준 가산점을 더한다.

```text
final_score = base_score + metadata_score
metadata_score = score_scale * (
    0.05 * category_match
  + 0.05 * matched_domain_fraction
)
```

불일치 후보도 제거하지 않으며 결과에는 `base_score`, `metadata_score`,
`category_match`, `matched_domains`가 남는다.

비교 실험을 위해 `fusion_method="normalized"`도 유지한다. 이 방식은 vector
점수에는 후보군 min-max, 0이 자연스러운 최솟값인 text 점수에는 max scaling을
적용한다.

## 파라미터

- `top_k`: 최종 반환 개수
- `candidate_k`: vector와 text 검색기가 각각 RRF에 전달할 후보 개수. 생략하면
  `max(top_k * 10, 50)`
- `rrf_k`: 상위와 하위 순위 차이를 완화하는 상수. 기본값 60
- `vector_weight`, `text_weight`: 두 검색기의 상대적 기여도. 내부에서 합이 1이
  되도록 조정
- `category`, `domains`: GLiNER2 metadata 가산점에 사용할 선택 조건
- `category_boost`, `domain_boost`: score scale 대비 가산점 비율. 기본값 각각 `0.05`

`candidate_k`는 데이터가 증가할 때 먼저 점검할 값이다. `rrf_k`는 전체 데이터
행 수에 맞춰 자동으로 키우지 않고 검색 평가 결과로 조정한다.

터미널에서도 실행할 수 있다.

```bash
python -m rag.retriever \
  --query "Claude" \
  --mode hybrid \
  --fusion rrf \
  --top-k 5 \
  --candidate-k 50 \
  --rrf-k 60 \
  --category 기술 \
  --domain AI
```

`--query`를 생략하면 터미널에서 직접 입력받는다.

## 기능 테스트 노트북

`rag/rag_function_test.ipynb`는 다음 순서로 구성되어 있다.

1. 프로젝트 경로와 `.env` 설정 확인
2. DB, pgvector 확장, 테이블 확인
3. 현재 `raw_articles`와 인덱스 건수 조회
4. 청킹 결과 미리보기
5. 선택적 HF embedding smoke test
6. 선택적 전체 기사 인덱싱
7. 저장 건수, 모델, 차원, L2 norm, orphan 확인
8. vector, text, RRF hybrid 검색 비교
9. 선택적 재인덱싱 안정성 확인
10. 사용자 직접 질의

외부 호출과 DB 쓰기를 실수로 실행하지 않도록 기본 플래그는 모두 `False`다.

```python
RUN_HF_SMOKE = False
RUN_INDEXING = False
RUN_SEARCH = False
RUN_REINDEX_CHECK = False
RUN_INTERACTIVE = False
```

권장 실행 순서는 다음과 같다.

1. 기본 상태로 DB와 chunk 미리보기 확인
2. `RUN_HF_SMOKE=True`로 embedding 한 건 확인
3. `RUN_INDEXING=True`로 현재 기사 저장
4. 다시 `False`로 바꾸고 `RUN_SEARCH=True`로 검색 비교
5. 필요할 때 `RUN_INTERACTIVE=True`로 직접 질의

노트북 source를 수정한 뒤 셀을 재실행하지 않으면 과거 출력이 남아 있을 수 있다.
특히 normalized hybrid에서 RRF로 변경한 뒤에는 검색 셀을 다시 실행해 rank와 RRF
기여도가 표시되는지 확인한다. 실행 결과를 기록으로 남기려면 셀 실행 후 노트북을
저장한다. 1024개 embedding 값 전체보다는 모델, 실제 차원, L2 norm을 기록하는 것이
읽기 쉽다.

## 검증

RAG 단위·DB 통합 테스트는 다음 명령으로 실행한다.

```bash
python -m unittest \
  rag.tests.test_chunk \
  rag.tests.test_embed \
  rag.tests.test_indexer \
  rag.tests.test_ingestion \
  rag.tests.test_topic_extractor \
  rag.tests.test_topic_indexer \
  rag.tests.test_event_extractor \
  rag.tests.test_event_indexer \
  rag.tests.test_retriever \
  rag.tests.test_db
```

검증 범위:

- tokenizer 유무에 따른 청킹과 overlap 경계
- HF 응답 개수·차원·정규화 검증
- vector/text 검색 SQL
- weighted RRF, 동점 dense rank, normalized 비교 방식
- Category/Domain 가산점과 후보 재정렬
- 재인덱싱 unique key와 DB FK/차원 제약
- Entity JSON 검증과 article_topics UPSERT/FK CASCADE
- 구조화 Event taxonomy, span 복원, Entity 일치, relation 경쟁과 중복 제거
- Event/Argument/Status 저장, 0-event 완료, 재처리와 FK CASCADE

테스트의 HF 호출은 mock이며 실제 API 품질이나 네트워크 상태를 검증하지 않는다.
DB 통합 테스트는 `https://test.invalid/` 접두사의 임시 기사만 만들고 종료 시
삭제한다.

2026-08-17 후속 구현에서 RAG test 62개가 통과했다. 그중 마지막 DB 비의존 중복수집
test를 추가하기 직전 local PostgreSQL 포함 61개도 전부 통과했다. 실제 GNews 1건을
수집해 본문 2개 chunk, BGE-M3 embedding, GLiNER2
`기술`/`AI·스타트업` metadata와 구조화 Event 5개를 저장했고, boost retrieval에서
metadata 가산점이 적용되는 것을 확인했다. smoke article과 모든 파생 행은 검증 후 삭제했다.

## 현재 제약

- 본문 원문과 body/fallback 상태는 `raw_articles`에 영구 저장하지 않는다.
- text 검색의 `'simple'` 설정은 한국어 형태소나 의미 유사도를 처리하지 않는다.
- 검색 결과는 chunk 단위이며 같은 기사의 여러 chunk를 묶는 로직은 아직 없다.
- vector/text 검색용 HNSW·GIN index는 아직 없다.
- digest 파이프라인과 뉴스 agent Tool에는 아직 통합하지 않았다.

## 향후 고도화

1. 검색 평가셋
   - 대표 질의와 관련 기사 정답을 축적한다.
   - vector, text, normalized hybrid, RRF를 Recall@k, Hit@k, nDCG@k로 비교한다.
   - `candidate_k`, `rrf_k`, 가중치는 평가셋으로 조정한다.
2. 기사 단위 결과 집계
   - 긴 기사에서 여러 chunk가 상위를 독점하지 않도록 `article_id`별 최고 chunk와
     추가 근거 chunk를 묶는다.
   - 필요하면 출처 다양성도 후처리에 반영한다.
3. text 검색 개선
   - 실제 corpus 언어를 확인하고 한국어·다국어 tokenizer 또는 검색 방식을
     선택한다.
   - 제목과 본문에 서로 다른 가중치를 주는 방식을 비교한다.
4. 대용량 검색 인덱스
   - embedding 검색에는 pgvector HNSW를 검토한다.
   - text 검색에는 미리 계산한 `tsvector` 컬럼과 GIN index를 검토한다.
   - embedding model별 데이터와 index를 분리해 다른 모델의 벡터를 섞지 않는다.
5. 뉴스 검색 조건
   - RRF 전에 `published_at`, 언어, 출처 같은 필터를 적용한다.
   - 최신성은 검색 범위 필터와 별도 rank signal 중 어느 쪽이 적합한지 평가한다.
6. 운영 관찰
   - query별 후보 수, 각 검색기 순위, RRF 기여도, latency를 기록한다.
   - 신규 기사에 embedding이 누락되지 않았는지 주기적으로 확인한다.
7. agent Tool 통합
   - `retrieve()`를 얇게 감싸 query, 기간, top-k를 받는 Tool 계약을 정의한다.
   - 검색 결과의 기사 URL과 근거 chunk를 agent 응답에 함께 전달한다.
   - 인덱싱은 기사 저장 이후 ingestion 단계 또는 별도 batch job으로 분리한다.

RRF는 각 검색기가 후보로 가져온 문서만 결합할 수 있다. 데이터가 늘었을 때 최종
품질이 떨어지면 RRF 식보다 먼저 vector/text 후보 Recall과 `candidate_k`를 확인한다.
