CREATE TABLE pages(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

    -- Whether the page is draft
    is_draft BOOLEAN NOT NULL,

    -- The user who created the page
    user_id INTEGER NOT NULL,

    -- Array of relevant questions, in format of `String`
    relevant_questions BLOB,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE page_sections(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,

    order_index INTEGER NOT NULL,

    title TEXT NOT NULL,
    content TEXT NOT NULL,

    -- TODO: link or updated image
    media BLOB,

    -- TODO: user input source
    client_sources BLOB,

    -- TODO: json format
    sources BLOB,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    CONSTRAINT `page_id_order_index` UNIQUE(page_id, order_index)
);
