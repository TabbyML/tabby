DROP TABLE github_repository_provider;

CREATE TABLE github_repository_provider(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    access_token TEXT,
    synced_at TIMESTAMP
);
