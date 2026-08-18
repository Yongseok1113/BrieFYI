-- RAG/VectorDB 스키마.
-- db/schema.sql의 기존 테이블을 먼저 만든 뒤 적용한다.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS article_chunks (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL
        REFERENCES raw_articles(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    chunk_text TEXT NOT NULL,
    UNIQUE (article_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL
        REFERENCES article_chunks(id) ON DELETE CASCADE,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    embedding VECTOR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (chunk_id, embedding_model),
    CHECK (vector_dims(embedding) = embedding_dimension)
);

CREATE TABLE IF NOT EXISTS article_topics (
    article_id  BIGINT PRIMARY KEY
                REFERENCES raw_articles(id) ON DELETE CASCADE,

    category    TEXT,
    domains     TEXT[],
    entities    TEXT[],

    topic_text  TEXT,
    embedding   vector(1024),

    created_at  TIMESTAMPTZ DEFAULT now()
);

-- GLiNER2가 추출한 구조화 Event. 현재 결과만 유지하며 Event embedding은 후속 범위다.
CREATE TABLE IF NOT EXISTS article_events (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL
        REFERENCES raw_articles(id) ON DELETE CASCADE,
    source_chunk_id BIGINT NOT NULL
        REFERENCES article_chunks(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    extractor_model TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    event_fingerprint TEXT NOT NULL,
    evidence_start INTEGER NOT NULL CHECK (evidence_start >= 0),
    evidence_end INTEGER NOT NULL CHECK (evidence_end > evidence_start),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (article_id, event_fingerprint)
);

CREATE TABLE IF NOT EXISTS article_event_arguments (
    event_id BIGINT NOT NULL
        REFERENCES article_events(id) ON DELETE CASCADE,
    argument_index INTEGER NOT NULL CHECK (argument_index >= 0),
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    span_start INTEGER NOT NULL CHECK (span_start >= 0),
    span_end INTEGER NOT NULL CHECK (span_end > span_start),

    PRIMARY KEY (event_id, argument_index)
);

CREATE TABLE IF NOT EXISTS article_event_index_status (
    article_id BIGINT PRIMARY KEY
        REFERENCES raw_articles(id) ON DELETE CASCADE,
    extractor_model TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    event_count INTEGER NOT NULL CHECK (event_count >= 0),
    error TEXT,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS article_chunks_article_id_idx
    ON article_chunks (article_id, chunk_index);

CREATE INDEX IF NOT EXISTS article_events_article_id_idx
    ON article_events (article_id, event_type);

CREATE INDEX IF NOT EXISTS article_event_arguments_normalized_text_idx
    ON article_event_arguments (normalized_text);

-- 검색 결과 요약 + grounding 검증 + LLM judge 재시도 루프의 시도별 기록
-- (rag_latest/summarize_agent.py). run_id로 한 번의 호출에 속한 여러 attempt를 묶는다.
CREATE TABLE IF NOT EXISTS summarize_agent_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    query TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    summary TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]',
    grounding_passed BOOLEAN NOT NULL,
    grounding_issues JSONB NOT NULL DEFAULT '[]',
    judge_score DOUBLE PRECISION NOT NULL CHECK (judge_score BETWEEN 0 AND 100),
    judge_reasoning TEXT,
    provider TEXT NOT NULL,
    passed_threshold BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS summarize_agent_runs_run_id_idx ON summarize_agent_runs (run_id);
