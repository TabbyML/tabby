CREATE TABLE github_repository_provider(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    application_id TEXT NOT NULL,
    secret TEXT NOT NULL,
    access_token TEXT,
    CONSTRAINT `idx_application_id` UNIQUE (`application_id`)
);
