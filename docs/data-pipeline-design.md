# data_pipeline 설계 — 프롬프트 엔지니어링 기반 학습 데이터 생성

`finetune/`이 요약 모델을 LoRA로 학습시키려면 (원문, 정답) 쌍이 충분히 필요하다. `finetune/src/summarize_ft/sources/digests_export.py`는 라이브 파이프라인이 Claude로 만든 `digests`를 재사용하지만, 볼륨을 늘리고 category/domain/entity/event 같은 풍부한 메타데이터까지 확보하려면 별도의 배치 파이프라인이 필요하다. 이 문서는 그 파이프라인(`data_pipeline/`)의 설계를 다룬다.

핵심 결정: **파인튜닝되지 않은 기본 오픈모델**(기본값: Groq `llama-3.3-70b-versatile`, 카드 등록 없는 무료 티어)에 프롬프트 엔지니어링으로 학습 데이터를 만든다. Claude API 비용/약관 이슈를 피하면서 모델을 "이름으로" 호출한다. 처음엔 HF Inference Providers를 검토했으나 무료 계정 크레딧이 월 $0.10뿐이고 카드 등록 없인 라우팅 자체가 막혀 있어(§7 참고) Groq로 전환했다 — `llm_client.py`에 `hf`/`anthropic` 경로도 남아 있어 `DATA_PIPELINE_LLM_PROVIDER`로 전환 가능하다.

## 1. 전체 흐름

```
raw_articles (pipeline_status='pending')
    │  변형1: extract.py
    │    - 구조화 소스(GNews)면 필드 그대로 매핑, 비구조화 소스면 LLM으로 추출
    │    - 키워드 추출은 로컬(KeyBERT, API 호출 아님)
    ▼
raw_articles (pipeline_status='extracted', keywords 채워짐)
    │  변형2: enrich.py (LLM)
    │    - insights, implications, category, domain, entity, event 원시값 생성
    ▼
(메모리/임시 상태, DB에 즉시 안 쓰고 normalize로 바로 전달하거나 enrichment에 raw_* 필드로 임시 저장)
    │  변형3: normalize.py
    │    - synonym_table과 fuzzy match 우선 시도, 신뢰도 낮으면만 LLM 호출
    ▼
enrichment 테이블 (raw_article_id FK, 정규화된 최종 값)
raw_articles (pipeline_status='normalized')

synonym_table: enrich.py가 쌓은 raw_category/domain/entity/event 값들을
synonym_builder.py가 주기적으로 임베딩 클러스터링해 자동 생성/갱신 (4절)
```

`pipeline_status`는 `raw_articles`에 컬럼으로 추가했다(`pending → extracted → enriched → normalized`, 실패 시 `failed`). 코랩/무료 API처럼 중간에 끊길 수 있는 환경을 전제로, 각 단계는 이 컬럼을 기준으로 "어디까지 됐는지"를 판단해 재시작 시 중복 처리 없이 이어서 진행한다.

## 2. 변형1 — 수집 + 추출 + 키워드

기존 `raw_articles` 필드 매핑(title/description/source/published_at/url)은 그대로 유지한다. GNews처럼 구조화된 소스는 API 응답을 그대로 매핑하고 LLM을 호출하지 않는다. 향후 스크래핑처럼 비구조화 소스가 추가될 경우를 대비해, 소스마다 `is_structured: bool`을 선언하게 하고(`sources/base.py`), `is_structured=False`인 소스만 LLM 추출 경로를 타도록 파이프라인에 파라미터로 흘려보낸다.

**키워드 추출은 LLM을 쓰지 않는다.** 벡터 검색/필터링에 쓸 키워드이므로, 로컬에서 도는 통계·임베딩 기반 방법(KeyBERT, 다국어 문장 임베딩 모델 `paraphrase-multilingual-MiniLM-L12-v2` 사용)으로 처리한다. API 호출이 아니라 요청 제한과 무관하고, 비용도 0이다. 참고로 벡터 DB에 넣을 임베딩 자체는 보통 키워드가 아니라 title+description 전체를 임베딩하는 것이 표준이라, 키워드는 "의미 검색용"이 아니라 "필터/보조 인덱스용" 메타데이터로 취급한다.

## 3. 변형2 — 팩트 해석/분류 메타데이터 생성 (LLM)

기본 모델에 프롬프트로 아래 항목을 생성시킨다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| insights | array | 팩트 해석 인사이트 3~5개 (기존 `tools/insight.py`와 동일한 성격) |
| implications | array | 비즈니스 시사점 |
| category | string | 대분류 (경제/기술/산업/금융 등) — 기사당 주 카테고리 하나 |
| domain | array | 세부 도메인 (반도체/AI/2차전지 등, 복수 가능) |
| entity | array | 관련 기업/기관 (삼성, NVIDIA, OpenAI 등, 복수 가능) |
| event | array | 이벤트 유형 (전쟁/사고/투자/M&A/실적/규제 등, 복수 가능) |

entity/event/domain은 기사 하나가 여러 개에 걸치는 경우가 흔해 배열로 둔다. category는 대분류 성격이라 단일값으로 둔다. 이 단계의 출력은 아직 원시값(raw)이다 — 같은 대상이 "삼성"/"삼성전자"/"Samsung Electronics"처럼 다르게 나올 수 있고, 이걸 정규화하는 것이 4·5절의 역할이다.

## 4. 통합 단어 테이블 (synonym_table)

category/domain/entity/event 네 개 차원(dimension)마다 원시값들을 모아 canonical value로 묶는 테이블이다. **최초 버전은 전량 자동 생성**한다(사람 검수 없이): 같은 dimension 안에서 원시값들을 다국어 문장 임베딩으로 벡터화하고, 코사인 유사도 기반 클러스터링(예: 임계값 이상인 것끼리 묶는 threshold clustering)으로 그룹을 만든 뒤, 그룹에서 가장 자주 등장한 표현을 canonical_value로 채택하고 나머지는 aliases에 넣는다. 이 자동 생성은 노이즈(비슷하지만 실제로는 다른 개념이 잘못 묶이는 경우)를 감수하는 트레이드오프이며, 검수는 파이프라인을 막지 않는 별도 스크립트(`scripts/run_synonym_builder.py --review`)로 나중에 진행한다. `synonym_table.reviewed` 컬럼으로 검수 여부를 추적한다.

## 5. 변형3 — 정규화 (하이브리드 fuzzy + LLM)

변형2의 원시값을 synonym_table의 canonical_value로 매핑한다. 매 항목마다 LLM을 부르면 요청 제한을 3배로 소모하므로, 다음 순서를 따른다.

1. **exact match**: 원시값이 이미 canonical_value거나 alias 목록에 있으면 즉시 채택.
2. **fuzzy match**: `rapidfuzz`로 canonical_value/alias들과의 유사도를 계산해 임계값 이상이면 채택.
3. **LLM fallback**: 1·2에서 못 찾았을 때만 LLM을 호출하되, **synonym_table에 있는 canonical_value 목록을 프롬프트에 명시적으로 제공하고 그중에서 고르게** 한다(자유 생성 금지 — 목록에 없는 새 카테고리를 모델이 지어내는 것을 막기 위함). 그래도 목록에 없는 완전히 새로운 개념이면 신규 canonical_value 후보로 반환하게 하고, `synonym_table`에 `reviewed=false`로 추가한다.

`enrichment.normalization_method`에 어떤 경로로 정규화됐는지(`exact`/`fuzzy`/`llm`) 기록해 나중에 품질을 추적할 수 있게 한다.

## 6. 저장소 설계

`db/schema.sql`에 반영했다.

- `raw_articles`: 기존 컬럼 유지 + `pipeline_status`(`pending`/`extracted`/`enriched`/`normalized`/`failed`), `keywords`(JSONB) 추가.
- `enrichment`: 신규 테이블. `raw_article_id`로 `raw_articles`를 FK 참조. 정규화된 최종값(`insights`/`implications`/`category`/`domain`/`entity`/`event`)과, 재처리 판단에 쓰는 원시값(`raw_category` 등) 및 계보 컬럼(`model_used`/`prompt_version`/`synonym_table_version`/`normalization_method`/`quality_flag`)을 함께 둔다. `digests`와는 별개 테이블이다 — `digests`는 라이브 파이프라인의 프로덕션 산출물, `enrichment`는 학습 데이터 생성용 산출물로 계보가 다르다.
- `synonym_table`: dimension별 canonical_value/aliases, `reviewed` 플래그.

`raw_articles`는 라이브 파이프라인(`main.py`)과 공유하는 테이블이라, 새 컬럼은 `DEFAULT`를 둬서 기존 insert 경로(`db/db.py:insert_articles`)가 이 컬럼을 몰라도 깨지지 않게 했다.

## 7. 요청 제한 준수

Groq 무료 티어는 카드 등록 없이 30 req/min · 6,000 tokens/min · 14,400 req/day 한도를 준다. `rate_limiter.py`가 설정된 요청/시간 한도(기본값: 25 req/60s, 안전 마진 포함)를 지키는 슬라이딩 윈도우 리미터를 제공하고, 모든 LLM 호출(`llm_client.py`)이 이를 거쳐 나가도록 한다. 429 응답은 지수 백오프로 재시도하고, 한도 초과가 계속되면 그 배치를 중단하고 `pipeline_status`를 그대로 둔 채(진행된 만큼만 반영) 다음 실행에서 이어가게 한다. rate_limiter는 요청 수만 세고 토큰 수는 안 세므로, 응답이 긴 프롬프트를 대량으로 돌릴 땐 TPM(6,000/min) 한도에 먼저 걸릴 수 있다는 점은 감안할 것.

LLM 호출은 최대 2단계(변형2, 그리고 변형3의 fallback)로 억제된다 — 변형1은 구조화 소스면 호출이 없고, 정규화는 fuzzy match가 대부분 처리해 LLM은 못 찾은 것만 부른다.

## 8. 컨테이너화

`data_pipeline/`은 레포 루트의 `db/`를 import하지 않는다 — 별도 Docker 이미지로 빌드되므로, DB 접근에 필요한 코드를 `data_pipeline/src/data_pipeline/db.py`에 자체적으로 포함시켰다(레포 루트 `db/db.py`와 로직은 유사하지만 독립된 모듈). `docker-compose.yml`, `Dockerfile`도 `data_pipeline/` 안에 자체적으로 둬서, 이 디렉토리 하나만으로 이미지를 빌드하고 띄울 수 있게 했다. DB는 메인 앱과 같은 Postgres 인스턴스를 `DATABASE_URL`로 공유한다(같은 `raw_articles`/`digests`를 보는 게 설계 의도이므로).

## 9. 학습 데이터 export

`finetune/src/summarize_ft/sources/enrichment_export.py`를 추가해 `raw_articles`(변형1 결과) + `enrichment`(변형3 결과)를 조인, 기존 공통 스키마(JSONL)로 변환한다. `scripts/prepare_data.py --sources enrichment`로 기존 `digest_pipeline`/`aihub`/`dacon`과 동일하게 다룰 수 있다.

## 10. 로드맵 / 오픈 이슈

1단계: `db/schema.sql` 반영 + `data_pipeline/` 스캐폴드 + 로컬 스모크 테스트(가짜 데이터로 extract→enrich→normalize 전 구간 실행). 2단계: GNews 실 데이터로 소량(수십 건) 파일럿을 돌려 `enrichment` 결과 품질을 육안 검수. 3단계: `synonym_table` 자동 생성 결과를 사람이 검수(`--review`)하고 `reviewed=true`로 확정. 4단계: 볼륨을 늘려가며 `finetune/scripts/prepare_data.py --sources enrichment digest_pipeline`로 학습 데이터를 합치고 실제 QLoRA 학습에 투입.

미확정 사항: (a) fuzzy match 임계값과 클러스터링 임계값은 실제 데이터로 튜닝 필요, (b) HF 무료 티어의 실제 시간당 한도는 계정/모델별로 변동이 있어 `rate_limiter.py` 설정값은 파일럿 실행 중 조정 필요, (c) `entity`(기업/기관) 차원은 나중에 티커/사업자번호 같은 외부 식별자와 연결할 여지가 있으나 현재 범위 밖.
