CREATE TABLE pages(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

    -- The user who created the page
    author_id INTEGER NOT NULL,

    title TEXT,
    content TEXT,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    FOREIGN KEY(author_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE page_sections(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,

    title TEXT NOT NULL,
    content TEXT,

    position INTEGER NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    --- Ensure that the position is unique for each page
    CONSTRAINT `unique_page_id_position` UNIQUE (page_id, position),

    FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE CASCADE
);
