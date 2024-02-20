CREATE TABLE user_completions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER INTEGER NOT NULL,
    completion_id VARCHAR(255) NOT NULL,

    language VARCHAR(255) NOT NULL,

    views INTEGER NOT NULL DEFAULT 0,
    selects INTEGER NOT NULL DEFAULT 0,
    dismisses INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE INDEX user_completions_user_id_language_idx ON user_completions (user_id, language);
CREATE INDEX user_completions_completion_id_idx ON user_completions (completion_id);