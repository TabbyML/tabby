CREATE TABLE IF NOT EXISTS google_oauth_credential (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id     VARCHAR(256) NOT NULL,
    client_secret VARCHAR(64) NOT NULL,
    redirect_uri  VARCHAR(256) NOT NULL,
    created_at    TIMESTAMP DEFAULT (DATETIME('now')),
    updated_at    TIMESTAMP DEFAULT (DATETIME('now'))
);
