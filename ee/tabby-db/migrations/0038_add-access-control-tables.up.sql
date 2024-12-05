CREATE TABLE user_groups (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL,

  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),

  CONSTRAINT idx_unique_name UNIQUE (name)
);

CREATE TABLE user_group_memberships (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

  user_id INTEGER NOT NULL,
  user_group_id INTEGER NOT NULL,

  is_group_admin BOOLEAN NOT NULL DEFAULT FALSE,

  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),

  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (user_group_id) REFERENCES user_groups(id) ON DELETE CASCADE,

  CONSTRAINT idx_unique_user_id_user_group_id UNIQUE (user_id, user_group_id)
);

CREATE TABLE source_id_read_access_policies (
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

  -- source_id doesn't come with DB constraint, need individual garbage collection job
  source_id VARCHAR(255) NOT NULL,

  user_group_id INTEGER NOT NULL,

  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),

  FOREIGN KEY (user_group_id) REFERENCES user_groups(id) ON DELETE CASCADE,

  -- access_policy is unique per source_id and user_group_id
  CONSTRAINT idx_unique_source_id_user_group_id UNIQUE (source_id, user_group_id)
);