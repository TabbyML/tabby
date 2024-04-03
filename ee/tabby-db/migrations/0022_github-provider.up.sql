CREATE TABLE github_repository_provider(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    application_id TEXT NOT NULL,
    secret TEXT NOT NULL,
    CONSTRAINT `idx_application_id` UNIQUE (`application_id`)
);
CREATE TABLE github_provided_repositories(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    name TEXT NOT NULL,
    git_url TEXT NOT NULL,
    CONSTRAINT `idx_provider_id_candidate_id` UNIQUE (`candidate_id`, `provider_id`)
);
