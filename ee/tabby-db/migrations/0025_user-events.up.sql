CREATE TABLE user_events (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    completion_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    payload TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX idx_user_events_created_at ON user_events(created_at);
CREATE INDEX idx_user_events_completion_id ON user_events(completion_id);
