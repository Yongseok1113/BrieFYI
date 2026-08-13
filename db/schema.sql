-- 원시 기사 저장 (구현 항목 #2)
CREATE TABLE IF NOT EXISTS raw_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_date TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT NOT NULL UNIQUE,
    source TEXT,
    published_at TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 요약/인사이트/시사점 결과 저장 (구현 항목 #3, #4)
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_date TEXT NOT NULL,
    keyword TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    insight_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 배포 이력 (구현 항목 #6)
CREATE TABLE IF NOT EXISTS send_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_id INTEGER NOT NULL REFERENCES digests(id),
    channel TEXT NOT NULL,
    recipient TEXT,
    status TEXT NOT NULL,
    error TEXT,
    sent_at TEXT NOT NULL DEFAULT (datetime('now'))
);
