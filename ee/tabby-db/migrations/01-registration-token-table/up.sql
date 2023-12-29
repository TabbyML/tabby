CREATE TABLE registration_token (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP DEFAULT (DATETIME('now')),
    CONSTRAINT `idx_token` UNIQUE (`token`)
);