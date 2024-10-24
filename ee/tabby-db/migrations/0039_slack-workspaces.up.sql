CREATE TABLE slack_workspaces(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  workspace_name VARCHAR(255) NOT NULL,
  workspace_id TEXT NOT NULL,
  bot_token TEXT NOT NULL,
  channels TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  CONSTRAINT slack_workspace_unique UNIQUE(workspace_id)
);