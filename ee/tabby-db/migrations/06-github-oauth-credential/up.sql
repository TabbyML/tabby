CREATE TABLE github_oauth_credential (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id     VARCHAR(32) NOT NULL,
    client_secret VARCHAR(64) NOT NULL,
    active        BOOLEAN DEFAULT (1),
    created_at    TIMESTAMP DEFAULT (DATETIME('now')),
    updated_at    TIMESTAMP DEFAULT (DATETIME('now'))
);
