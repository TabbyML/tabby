CREATE TABLE integrations(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    display_name TEXT NOT NULL,
    access_token TEXT NOT NULL,
    api_base TEXT,
    error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    synced BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE provided_repositories(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    integration_id INTEGER NOT NULL,
    vendor_id TEXT NOT NULL,
    name TEXT NOT NULL,
    git_url TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (integration_id) REFERENCES integrations(id) ON DELETE CASCADE,
    CONSTRAINT idx_unique_integration_id_vendor_id UNIQUE (integration_id, vendor_id)
);

INSERT INTO integrations(kind, display_name, access_token)
    SELECT 'github', display_name, access_token FROM github_repository_provider WHERE access_token IS NOT NULL;

INSERT INTO integrations(kind, display_name, access_token)
    SELECT 'gitlab', display_name, access_token FROM gitlab_repository_provider WHERE access_token IS NOT NULL;

INSERT INTO provided_repositories(integration_id, vendor_id, name, git_url, active)
    SELECT integrations.id, vendor_id, github_provided_repositories.name, git_url, active FROM github_provided_repositories
    JOIN github_repository_provider ON github_repository_provider_id = github_repository_provider.id
    JOIN integrations ON kind = 'github' AND github_repository_provider.access_token = integrations.access_token;

INSERT INTO provided_repositories(integration_id, vendor_id, name, git_url, active)
    SELECT integrations.id, vendor_id, gitlab_provided_repositories.name, git_url, active FROM gitlab_provided_repositories
    JOIN gitlab_repository_provider ON gitlab_repository_provider_id = gitlab_repository_provider.id
    JOIN integrations ON kind = 'gitlab' AND gitlab_repository_provider.access_token = integrations.access_token;

DROP TABLE github_provided_repositories;
DROP TABLE gitlab_provided_repositories;
DROP TABLE github_repository_provider;
DROP TABLE gitlab_repository_provider;
