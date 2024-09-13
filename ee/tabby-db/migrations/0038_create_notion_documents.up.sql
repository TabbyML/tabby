-- Add up migration script here

CREATE TABLE notion_documents(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL,
  integration_id TEXT NOT NULL,
  integration_type TEXT CHECK(integration_type in ('database', 'page', 'wiki')) NOT NULL,
  access_token TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  CONSTRAINT idx_name UNIQUE(name),
  CONSTRAINT idx_integration_id UNIQUE(integration_id)
);