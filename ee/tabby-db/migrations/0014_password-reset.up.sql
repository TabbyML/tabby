CREATE TABLE password_reset(
    user_id INTEGER PRIMARY KEY NOT NULL,
    code VARCHAR(36) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now'))
);
