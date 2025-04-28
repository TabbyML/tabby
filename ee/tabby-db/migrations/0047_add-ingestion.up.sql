CREATE TABLE IF NOT EXISTS ingested_documents (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

    -- User-provided document source
    source TEXT NOT NULL,

    -- User-provided document ID, unique within the same source
    doc_id TEXT NOT NULL,

    link TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,

    -- Track progress of ingestion
    status TEXT NOT NULL CHECK (status IN ('pending', 'indexed', 'failed')),

    -- Expiration time in Unix timestamp (0 means never expired, should be cleaned by API)
    expired_at INTEGER NOT NULL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Enforce unique constraint on (source, doc_id) to ensure document IDs are unique within the same source
    UNIQUE (source, doc_id)
);
