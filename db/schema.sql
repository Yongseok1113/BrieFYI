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
