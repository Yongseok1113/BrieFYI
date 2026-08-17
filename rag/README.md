# BrieFYI RAG

현재 PostgreSQL의 `raw_articles`를 청킹하고 BGE-M3 embedding을 저장한 뒤,
vector, text 또는 hybrid 방식으로 검색한다. 외부 기사 데이터는 읽지 않는다.

현재 digest 파이프라인이나 `graph/pipeline.py`에는 연결하지 않았다. 검색 기능과
DB 계약을 먼저 검증한 뒤, 향후 뉴스 agent가 호출하는 Tool로 `retrieve()`를 감싸는
방향을 전제로 한다.

## 현재 구현 범위

| 파일 | 역할 |
| --- | --- |
| `chunk.py` | 기사 제목과 description을 결합하고 chunk 목록 생성 |
| `embed.py` | Hugging Face Inference API로 BGE-M3 dense embedding 생성 |
| `indexer.py` | 현재 `raw_articles`를 청킹·임베딩해 PostgreSQL에 저장 |
| `retriever.py` | vector, text, normalized hybrid, RRF hybrid 검색 |
| `rag_function_test.ipynb` | DB 확인부터 인덱싱·검색까지 단계별 기능 테스트 |

```text
raw_articles(title, description)
    -> build_article_text()
    -> split_text()
    -> article_chunks
    -> embed_texts(BAAI/bge-m3)
    -> chunk_embeddings

사용자 query
    -> vector search + text search
    -> weighted RRF
    -> 관련 chunk 목록
```

이 모듈의 인덱싱 입력은 현재 BrieFYI DB의 `raw_articles`로 제한한다.

## 설정과 실행 조건

프로젝트 루트의 `.env`에는 embedding과 4-Layer 추출에 사용할 API 키가 필요하다.

```dotenv
HF_TOKEN="your_huggingface_token"
HF_EMBEDDING_MODEL="BAAI/bge-m3"
ANTHROPIC_API_KEY="your_anthropic_api_key"
```

고정 가능한 값은 환경변수로 늘리지 않고 소스 설정을 사용한다.

- embedding dimension: `1024`
- HF 요청 timeout: `120`초
- chunk size: `500` token
- 현재 overlap: `0`

필요한 Python 의존성은 `anthropic`, `requests`, `psycopg`, `pgvector`, `python-dotenv`이며
프로젝트 `requirements.txt`에 포함되어 있다. DB는 pgvector 확장이 설치된
PostgreSQL 16이어야 한다.

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
```

`article_chunks`는 `(article_id, chunk_index)`, `chunk_embeddings`는
`(chunk_id, embedding_model)`을 unique key로 사용한다. embedding 행에는 모델 이름,
선언된 차원, 실제 vector를 함께 저장하며
`vector_dims(embedding) = embedding_dimension` check로 차원을 검증한다.

동일 chunk에 여러 모델의 벡터를 함께 둘 수 있고, 검색할 때는 query와 같은
`embedding_model`, `embedding_dimension`만 선택한다.

## 4-Layer metadata prototype

현재 baseline에서 4-Layer는 검색용 metadata이며 embedding 대상이 아니다.

```text
외부 입력: Category / Domain
                    + 기사 제목 / description
                              -> Anthropic Entity / Event 추출
                              -> article_topics UPSERT

raw_articles 1 --- 1 article_topics
     |
     +--- N article_chunks 1 --- N chunk_embeddings
```

- `category`: 기술, 경제 같은 대분류
- `domains`: AI, 반도체 같은 세부 도메인
- `entities`: 기사에서 핵심적으로 다루는 회사·기관·인물·제품, 최대 3개
- `events`: 핵심 사건·행동을 나타내는 짧은 명사구, 최대 2개

`topic_text`와 topic `embedding`은 후속 실험용 컬럼이며 현재는 NULL로 둔다. 기존
기사의 chunk embedding은 다시 만들지 않고 그대로 연결해서 사용한다.

현재 DB에서 내용이 확인된 AI 기사 10건을 독립 실행하려면 다음 명령을 사용한다.

```bash
python -m rag.topic_indexer \
  --article-ids 19 20 21 22 23 24 25 26 27 28 \
  --category 기술 \
  --domain AI
```

여러 Domain을 전달할 때는 `--domain`을 반복한다. CLI는 `init_db()`로 스키마를 먼저
확인하고, 지정한 ID가 하나라도 없으면 Anthropic API를 호출하기 전에 중단한다.
Category/Domain 검색 필터, Entity/Event boost, 전체 기사 백필, Collector와 pipeline
연결은 이 prototype에 포함하지 않는다.

## 청킹과 Embedding

현재 DB에는 기사 전문이 아니라 짧은 `title`과 `description`이 저장되어 있다.
로컬 tokenizer를 따로 다운로드하지 않는 HF API 방식이므로 현재 각 기사는 하나의
chunk가 된다. tokenizer가 전달되면 offset mapping을 이용한 fixed-token 청킹을
수행한다.

overlap 로직은 구현되어 있지만 짧은 원본이 과도하게 겹치는 것을 막기 위해 현재
0으로 비활성화했다.

```python
CHUNK_OVERLAP_TOKENS = 0

# 전문 기사 corpus가 준비되면 검토
# CHUNK_OVERLAP_TOKENS = 50
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
)
```

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
  --rrf-k 60
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
  rag.tests.test_topic_indexer \
  rag.tests.test_retriever \
  rag.tests.test_db
```

검증 범위:

- tokenizer 유무에 따른 청킹과 overlap 경계
- HF 응답 개수·차원·정규화 검증
- vector/text 검색 SQL
- weighted RRF, 동점 dense rank, normalized 비교 방식
- 재인덱싱 unique key와 DB FK/차원 제약
- Entity/Event JSON 검증과 article_topics UPSERT/FK CASCADE

테스트의 HF 호출은 mock이며 실제 API 품질이나 네트워크 상태를 검증하지 않는다.
DB 통합 테스트는 `https://test.invalid/` 접두사의 임시 기사만 만들고 종료 시
삭제한다.

## 현재 제약

- `raw_articles`에는 전문이 아니라 GNews의 제목과 description만 저장된다.
- 로컬 tokenizer가 없어 현재 각 기사는 한 chunk다.
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
