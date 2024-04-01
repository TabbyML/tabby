DROP TABLE IF EXISTS repository_meta;
CREATE TABLE repository_meta(
    git_url TEXT NOT NULL,
    filepath TEXT NOT NULL,
    language TEXT NOT NULL,
    max_line_length INTEGER NOT NULL,
    avg_line_length REAL NOT NULL,
    alphanum_fraction REAL NOT NULL,
    tags TEXT NOT NULL
);
