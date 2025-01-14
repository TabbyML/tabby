CREATE TABLE pages(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

    -- The user who created the page
    author_id INTEGER NOT NULL,

    title TEXT,
    summary TEXT,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    FOREIGN KEY(author_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE page_sections(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,

    position INTEGER NOT NULL,

    title TEXT NOT NULL,
    content TEXT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE,
    CONSTRAINT `page_id_position` UNIQUE(page_id, position)
);
