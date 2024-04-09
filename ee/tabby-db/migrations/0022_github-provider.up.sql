CREATE TABLE github_repository_provider(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    application_id TEXT NOT NULL,
    secret TEXT NOT NULL,
    access_token TEXT,
    CONSTRAINT `idx_application_id` UNIQUE (`application_id`)
);

CREATE TABLE github_provided_repositories(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    github_repository_provider_id INTEGER NOT NULL,
    -- vendor_id refers to the `node_id` field in the output of https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#list-organization-repositories
    vendor_id TEXT NOT NULL,
    name TEXT NOT NULL,
    git_url TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT `idx_vendor_id` UNIQUE (`vendor_id`),
    FOREIGN KEY (github_repository_provider_id) REFERENCES github_repository_provider(id) ON DELETE CASCADE
);
