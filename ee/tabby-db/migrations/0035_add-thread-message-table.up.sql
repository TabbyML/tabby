CREATE TABLE threads(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

    -- Whether the thread is ephemeral (e.g from chat sidebar)
    is_ephemeral BOOLEAN NOT NULL,

    -- The user who created the thread
    user_id INTEGER NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    -- Array of relevant questions, in format of `String`
    relevant_questions BLOB,

    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE thread_messages(
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL,

    role TEXT NOT NULL,
    content TEXT NOT NULL,

    -- Array of code attachments, in format of `ThreadMessageAttachmentCode`
    code_attachments BLOB,

    -- Array of client code attachments, in format of `ThreadMessageAttachmentClientCode`
    client_code_attachments BLOB,

    -- Array of doc attachments, in format of `ThreadMessageAttachmentDoc`
    doc_attachments BLOB,

    created_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),
    updated_at TIMESTAMP NOT NULL DEFAULT (DATETIME('now')),

    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
);