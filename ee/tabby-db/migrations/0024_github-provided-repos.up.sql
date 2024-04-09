CREATE TABLE github_provided_repositories(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    github_repository_provider_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    git_url TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    FOREIGN KEY (github_repository_provider_id) REFERENCES github_repository_provider(id),
    CONSTRAINT provided_repository_unique_name UNIQUE (name)
);
