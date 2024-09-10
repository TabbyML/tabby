CREATE TABLE _sqlx_migrations(
  version BIGINT PRIMARY KEY,
  description TEXT NOT NULL,
  installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  success BOOLEAN NOT NULL,
  checksum BLOB NOT NULL,
  execution_time BIGINT NOT NULL
);
CREATE TABLE registration_token(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  token VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP DEFAULT(DATETIME('now')),
  CONSTRAINT `idx_token` UNIQUE(`token`)
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE users(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email VARCHAR(150) NOT NULL COLLATE NOCASE,
  is_admin BOOLEAN NOT NULL DEFAULT 0,
  created_at TIMESTAMP DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP DEFAULT(DATETIME('now')),
  auth_token VARCHAR(128) NOT NULL,
  active BOOLEAN NOT NULL DEFAULT 1,
  password_encrypted VARCHAR(128),
  avatar BLOB DEFAULT NULL,
  name VARCHAR(255),
  CONSTRAINT `idx_email` UNIQUE(`email`)
  CONSTRAINT `idx_auth_token` UNIQUE(`auth_token`)
);
CREATE TABLE invitations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email VARCHAR(150) NOT NULL COLLATE NOCASE,
  code VARCHAR(36) NOT NULL,
  created_at TIMESTAMP DEFAULT(DATETIME('now')),
  CONSTRAINT `idx_email` UNIQUE(`email`)
  CONSTRAINT `idx_code` UNIQUE(`code`)
);
CREATE TABLE job_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job VARCHAR(255) NOT NULL,
  start_ts TIMESTAMP NOT NULL,
  end_ts TIMESTAMP,
  exit_code INTEGER,
  stdout TEXT NOT NULL,
  stderr TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP DEFAULT(DATETIME('now'))
  ,
  command TEXT,
  started_at TIMESTAMP
);
CREATE TABLE repositories(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL,
  git_url VARCHAR(255) NOT NULL,
  CONSTRAINT `idx_name` UNIQUE(`name`)
  CONSTRAINT `idx_git_url` UNIQUE(`git_url`)
);
CREATE TABLE server_setting(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  security_allowed_register_domain_list STRING,
  security_disable_client_side_telemetry BOOLEAN NOT NULL DEFAULT FALSE,
  network_external_url STRING NOT NULL DEFAULT 'http://localhost:8080'
  ,
  billing_enterprise_license STRING
);
CREATE TABLE email_setting(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  smtp_username VARCHAR(255) NOT NULL,
  smtp_password VARCHAR(255) NOT NULL,
  smtp_server VARCHAR(255) NOT NULL,
  from_address VARCHAR(255) NOT NULL,
  encryption VARCHAR(255) NOT NULL DEFAULT 'ssltls',
  auth_method VARCHAR(255) NOT NULL DEFAULT 'plain'
  ,
  smtp_port INTEGER NOT NULL DEFAULT 25
);
CREATE TABLE oauth_credential(
  id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  provider TEXT NOT NULL,
  client_id VARCHAR(256) NOT NULL,
  client_secret VARCHAR(64) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  CONSTRAINT `idx_provider` UNIQUE(`provider`)
);
CREATE TABLE user_completions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER INTEGER NOT NULL,
  completion_id VARCHAR(255) NOT NULL,
  language VARCHAR(255) NOT NULL,
  views INTEGER NOT NULL DEFAULT 0,
  selects INTEGER NOT NULL DEFAULT 0,
  dismisses INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX user_completions_completion_id_idx ON user_completions(
  completion_id
);
CREATE INDEX idx_job_created_at ON job_runs(job, created_at);
CREATE INDEX idx_repository_name ON repositories(name);
CREATE INDEX idx_user_completion_user_id_created_at_language ON user_completions(
  user_id,
  created_at,
  language
);
CREATE TABLE user_events(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  payload BLOB NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX idx_user_events_user_id ON user_events(user_id);
CREATE INDEX idx_user_events_created_at ON user_events(created_at);
CREATE TABLE refresh_tokens(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  token VARCHAR(255) NOT NULL COLLATE NOCASE,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT(DATETIME('now')),
  CONSTRAINT `idx_token` UNIQUE(`token`)
  FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE password_reset(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL UNIQUE,
  code VARCHAR(36) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE integrations(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  display_name TEXT NOT NULL,
  access_token TEXT NOT NULL,
  api_base TEXT,
  error TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  synced BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE provided_repositories(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  integration_id INTEGER NOT NULL,
  vendor_id TEXT NOT NULL,
  name TEXT NOT NULL,
  git_url TEXT NOT NULL,
  active BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  FOREIGN KEY(integration_id) REFERENCES integrations(id) ON DELETE CASCADE,
  CONSTRAINT idx_unique_integration_id_vendor_id UNIQUE(integration_id, vendor_id)
);
CREATE INDEX `idx_job_runs_command` ON job_runs(command);
CREATE TABLE threads(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  -- Whether the thread is ephemeral(e.g from chat sidebar)
  is_ephemeral BOOLEAN NOT NULL,
  -- The user who created the thread
  user_id INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
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
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
);
CREATE TABLE web_documents(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL,
  url TEXT NOT NULL,
  is_preset BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  CONSTRAINT idx_name UNIQUE(name),
  CONSTRAINT idx_url UNIQUE(url)
);
CREATE TABLE user_groups(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  CONSTRAINT idx_unique_name UNIQUE(name)
);
CREATE TABLE user_group_memberships(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  user_group_id INTEGER NOT NULL,
  is_group_admin BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY(user_group_id) REFERENCES user_groups(id) ON DELETE CASCADE,
  CONSTRAINT idx_unique_user_id_user_group_id UNIQUE(user_id, user_group_id)
);
CREATE TABLE source_id_read_access_policies(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  -- source_id doesn't come with DB constraint, need individual garbage collection job
source_id VARCHAR(255) NOT NULL,
user_group_id INTEGER NOT NULL,
created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
FOREIGN KEY(user_group_id) REFERENCES user_groups(id) ON DELETE CASCADE,
-- access_policy is unique per source_id and user_group_id
  CONSTRAINT idx_unique_source_id_user_group_id UNIQUE(source_id, user_group_id)
);
