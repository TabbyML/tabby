CREATE TABLE github_provided_repositories(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    github_repository_provider_id INTEGER NOT NULL,
    -- vendor_id from https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#list-repositories-for-a-user
    vendor_id TEXT NOT NULL,
    name TEXT NOT NULL,
    git_url TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (github_repository_provider_id) REFERENCES github_repository_provider(id) ON DELETE CASCADE,
    CONSTRAINT `idx_vendor_id_provider_id` UNIQUE (vendor_id, github_repository_provider_id)
);
CREATE INDEX github_provided_repositories_updated_at ON github_provided_repositories(updated_at);
