CREATE TABLE github_provider (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    github_url TEXT NOT NULL,
    application_id TEXT NOT NULL,
    secret TEXT NOT NULL,
    CONSTRAINT `idx_name` UNIQUE (`name`)
);
