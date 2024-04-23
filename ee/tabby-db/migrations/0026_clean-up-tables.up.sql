DROP TABLE github_oauth_credential;
DROP TABLE google_oauth_credential;

-- Drop the entire refresh_token table to resolve the foreign key issue.
-- This is acceptable as it is equivalent to asking all users to sign in again for the new release.
DROP TABLE refresh_tokens;

CREATE TABLE refresh_tokens(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  token VARCHAR(255) NOT NULL COLLATE NOCASE,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT(DATETIME('now')),
  CONSTRAINT `idx_token` UNIQUE(`token`)

  FOREIGN KEY(user_id) REFERENCES users(id)
);