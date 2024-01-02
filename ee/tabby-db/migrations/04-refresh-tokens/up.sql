CREATE TABLE refresh_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token VARCHAR(255) NOT NULL COLLATE NOCASE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT (DATETIME('now')),
    CONSTRAINT `idx_token` UNIQUE (`token`)
);