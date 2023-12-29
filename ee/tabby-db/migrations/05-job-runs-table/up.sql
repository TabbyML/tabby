CREATE TABLE job_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job        VARCHAR(255) NOT NULL,
    start_ts   TIMESTAMP NOT NULL,
    end_ts     TIMESTAMP,
    exit_code  INTEGER,
    stdout     TEXT NOT NULL,
    stderr     TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP DEFAULT (DATETIME('now'))
);