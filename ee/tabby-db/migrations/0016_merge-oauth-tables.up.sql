CREATE TABLE oauth_credential (
    id            INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    provider      TEXT NOT NULL,
    client_id     VARCHAR(256) NOT NULL,
    client_secret VARCHAR(64) NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at    TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    CONSTRAINT `idx_provider` UNIQUE (`provider`)
);

INSERT INTO oauth_credential(provider, client_id, client_secret, created_at, updated_at)
    SELECT 'google', client_id, client_secret, created_at, updated_at FROM google_oauth_credential;

INSERT INTO oauth_credential(provider, client_id, client_secret, created_at, updated_at)
    SELECT 'github', client_id, client_secret, created_at, updated_at FROM github_oauth_credential;