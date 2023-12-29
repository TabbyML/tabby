CREATE TABLE users (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    email              VARCHAR(150) NOT NULL COLLATE NOCASE,
    password_encrypted VARCHAR(128) NOT NULL,
    is_admin           BOOLEAN NOT NULL DEFAULT 0,
    created_at         TIMESTAMP DEFAULT (DATETIME('now')),
    updated_at         TIMESTAMP DEFAULT (DATETIME('now')),
    auth_token         VARCHAR(128) NOT NULL,

    CONSTRAINT `idx_email`      UNIQUE (`email`)
    CONSTRAINT `idx_auth_token` UNIQUE (`auth_token`)
);