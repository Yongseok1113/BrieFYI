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
    events      TEXT[],

    topic_text  TEXT,
    embedding   vector(1024),

    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS article_chunks_article_id_idx
    ON article_chunks (article_id, chunk_index);
