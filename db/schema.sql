-- PostgreSQL 스키마. 두 곳에서 쓰인다.
--   1) Dockerfile.db: 컨테이너 최초 기동 시 /docker-entrypoint-initdb.d 로 자동 실행
--   2) db/db.py init_db(): 앱 시작 시 idempotent 하게 재적용 (IF NOT EXISTS)

-- 원시 기사 저장 (구현 항목 #2)
CREATE TABLE IF NOT EXISTS raw_articles (
    id BIGSERIAL PRIMARY KEY,
    digest_date DATE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT NOT NULL UNIQUE,
    source TEXT,
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS raw_articles_digest_date_idx ON raw_articles (digest_date);

-- data_pipeline/ (학습 데이터 생성용 프롬프트 엔지니어링 파이프라인, docs/data-pipeline-design.md 참고)이
-- 변형1(추출+키워드) 진행 상태를 추적하기 위해 쓰는 컬럼. 기존 라이브 파이프라인(main.py)의
-- insert_articles()는 이 컬럼을 모르고 그대로 두면 DEFAULT 'pending'이 채워진다.
ALTER TABLE raw_articles ADD COLUMN IF NOT EXISTS pipeline_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE raw_articles ADD COLUMN IF NOT EXISTS keywords JSONB;

CREATE INDEX IF NOT EXISTS raw_articles_pipeline_status_idx ON raw_articles (pipeline_status);

-- 변형2+변형3 결과 (팩트 해석/분류 메타데이터). digests와는 별개 테이블이다 —
-- digests는 라이브 파이프라인이 Claude로 만드는 프로덕션 산출물이고, 이 테이블은
-- data_pipeline이 학습 데이터 생성을 위해 프롬프트 엔지니어링으로 만드는 산출물이라
-- 목적/계보(lineage)가 다르다.
CREATE TABLE IF NOT EXISTS enrichment (
    id BIGSERIAL PRIMARY KEY,
    raw_article_id BIGINT NOT NULL REFERENCES raw_articles (id) ON DELETE CASCADE,

    -- 정규화(변형3) 완료된 최종 값
    insights JSONB NOT NULL DEFAULT '[]',
    implications JSONB NOT NULL DEFAULT '[]',
    category TEXT,
    domain JSONB NOT NULL DEFAULT '[]',
    entity JSONB NOT NULL DEFAULT '[]',
    event JSONB NOT NULL DEFAULT '[]',

    -- 정규화 이전(변형2) 원시값. 통합 단어 테이블이 나중에 바뀌었을 때
    -- 무엇을 다시 정규화해야 하는지 판단하는 데 쓰인다.
    raw_category TEXT,
    raw_domain JSONB,
    raw_entity JSONB,
    raw_event JSONB,

    normalization_method TEXT,       -- 'exact' | 'fuzzy' | 'llm'
    model_used TEXT,
    prompt_version TEXT,
    synonym_table_version TEXT,
    quality_flag TEXT NOT NULL DEFAULT 'unverified',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS enrichment_raw_article_id_idx ON enrichment (raw_article_id);

-- category/domain/entity/event 이음동의어 통합 테이블 (design doc 4절).
-- 최초 버전은 자동 생성(임베딩 클러스터링)되고, 사람 검수는 별도 스크립트로 나중에 진행한다.
CREATE TABLE IF NOT EXISTS synonym_table (
    id BIGSERIAL PRIMARY KEY,
    dimension TEXT NOT NULL,             -- 'category' | 'domain' | 'entity' | 'event'
    canonical_value TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]',
    reviewed BOOLEAN NOT NULL DEFAULT false,  -- 사람이 검수했는지 여부 (추후 검수 워크플로용)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dimension, canonical_value)
);

CREATE INDEX IF NOT EXISTS synonym_table_dimension_idx ON synonym_table (dimension);

-- 요약/인사이트/시사점 결과 저장 (구현 항목 #3, #4)
CREATE TABLE IF NOT EXISTS digests (
    id BIGSERIAL PRIMARY KEY,
    digest_date DATE NOT NULL,
    keyword TEXT NOT NULL,
    summary_json JSONB NOT NULL,
    insight_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS digests_digest_date_idx ON digests (digest_date);

-- 배포 이력 (구현 항목 #6)
CREATE TABLE IF NOT EXISTS send_log (
    id BIGSERIAL PRIMARY KEY,
    digest_id BIGINT NOT NULL REFERENCES digests (id) ON DELETE CASCADE,
    channel TEXT NOT NULL,
    recipient TEXT,
    status TEXT NOT NULL,
    error TEXT,
    sent_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS send_log_digest_id_idx ON send_log (digest_id);
