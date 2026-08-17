-- RAG / VectorDB schema
-- 기존 db/schema.sql에서 raw_articles 테이블이 먼저 생성되어 있어야 한다.

CREATE EXTENSION IF NOT EXISTS vector;


-- =========================================================
-- 1. Article Topics
-- 기사 하나당 하나의 분류 정보
-- =========================================================

CREATE TABLE IF NOT EXISTS article_topics (
    article_id BIGINT PRIMARY KEY
        REFERENCES raw_articles(id) ON DELETE CASCADE,

    category TEXT,
    domains TEXT[],

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- =========================================================
-- 2. Article Entities
-- 기사 하나에서 여러 entity 추출 가능
-- 예: NVIDIA, Jensen Huang, Blackwell
-- =========================================================

CREATE TABLE IF NOT EXISTS article_entities (
    id BIGSERIAL PRIMARY KEY,

    article_id BIGINT NOT NULL
        REFERENCES raw_articles(id) ON DELETE CASCADE,

    entity_text TEXT NOT NULL,
    entity_type TEXT,

    embedding_model TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (article_id, entity_text)
);


-- =========================================================
-- 3. Article Events
-- 기사 하나에서 여러 event 추출 가능
-- 예: "NVIDIA가 Blackwell Ultra를 공개"
-- =========================================================

CREATE TABLE IF NOT EXISTS article_events (
    id BIGSERIAL PRIMARY KEY,

    article_id BIGINT NOT NULL
        REFERENCES raw_articles(id) ON DELETE CASCADE,

    event_text TEXT NOT NULL,
    event_type TEXT,

    embedding_model TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (article_id, event_text)
);


-- =========================================================
-- 4. Article Chunks
-- 원문 기사 1개 → 여러 chunk
-- =========================================================

CREATE TABLE IF NOT EXISTS article_chunks (
    id BIGSERIAL PRIMARY KEY,

    article_id BIGINT NOT NULL
        REFERENCES raw_articles(id) ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL
        CHECK (chunk_index >= 0),

    chunk_text TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (article_id, chunk_index)
);


-- =========================================================
-- 5. Chunk Embeddings
-- chunk와 embedding을 분리해 embedding model 변경/비교 가능
-- =========================================================

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id BIGSERIAL PRIMARY KEY,

    chunk_id BIGINT NOT NULL
        REFERENCES article_chunks(id) ON DELETE CASCADE,

    embedding_model TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (chunk_id, embedding_model)
);


-- =========================================================
-- Indexes
-- =========================================================

CREATE INDEX IF NOT EXISTS article_topics_category_idx
    ON article_topics (category);

CREATE INDEX IF NOT EXISTS article_entities_article_id_idx
    ON article_entities (article_id);

CREATE INDEX IF NOT EXISTS article_events_article_id_idx
    ON article_events (article_id);

CREATE INDEX IF NOT EXISTS article_chunks_article_id_idx
    ON article_chunks (article_id, chunk_index);

CREATE INDEX IF NOT EXISTS chunk_embeddings_chunk_id_idx
    ON chunk_embeddings (chunk_id);