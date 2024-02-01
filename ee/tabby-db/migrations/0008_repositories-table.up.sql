CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    git_url VARCHAR(255) NOT NULL,
    CONSTRAINT `idx_name` UNIQUE (`name`)
    CONSTRAINT `idx_git_url` UNIQUE (`git_url`)
);
