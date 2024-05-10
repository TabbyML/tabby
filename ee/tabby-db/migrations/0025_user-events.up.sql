CREATE TABLE user_events (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    payload BLOB NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX idx_user_events_user_id ON user_events(user_id);
CREATE INDEX idx_user_events_created_at ON user_events(created_at);
