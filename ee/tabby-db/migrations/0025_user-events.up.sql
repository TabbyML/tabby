CREATE TABLE user_events (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    payload BLOB NOT NULL,
    FOREIGN KEY (user_name) REFERENCES users(email) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX idx_user_events_created_at ON user_events(created_at);
