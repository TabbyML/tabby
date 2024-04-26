CREATE TABLE gitlab_repository_provider(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    access_token TEXT,
    synced_at TIMESTAMP
);

CREATE TABLE gitlab_provided_repositories(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    gitlab_repository_provider_id INTEGER NOT NULL,
    -- vendor_id from https://docs.gitlab.com/ee/api/repositories.html
    vendor_id TEXT NOT NULL,
    name TEXT NOT NULL,
    git_url TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (gitlab_repository_provider_id) REFERENCES gitlab_repository_provider(id) ON DELETE CASCADE,
    CONSTRAINT `idx_vendor_id_provider_id` UNIQUE (vendor_id, gitlab_repository_provider_id)
);
CREATE INDEX gitlab_provided_repositories_updated_at ON gitlab_provided_repositories(updated_at);
