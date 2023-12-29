CREATE TABLE invitations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    email              VARCHAR(150) NOT NULL COLLATE NOCASE,
    code               VARCHAR(36) NOT NULL,
    created_at         TIMESTAMP DEFAULT (DATETIME('now')),
    CONSTRAINT `idx_email` UNIQUE (`email`)
    CONSTRAINT `idx_code`  UNIQUE (`code`)
);