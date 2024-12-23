CREATE TABLE ldap_credential(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL,
  host STRING NOT NULL,
  smtp_port INTEGER NOT NULL DEFAULT 389,
  bind_dn STRING NOT NULL,
  bind_password STRING NOT NULL,
  base_dn STRING NOT NULL,
  user_filter STRING NOT NULL,
  encryption STRING NOT NULL DEFAULT 'none',
  skip_tls_verify BOOLEAN NOT NULL DEFAULT FALSE,
  --- the attribute to be used as the Tabby user email address
  email_attribute STRING NOT NULL DEFAULT 'email',
  --- the attribute to be used as the Tabby user name
  name_attribute STRING NOT NULL DEFAULT 'name',
  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  --- name is unique to distinguish different LDAP configurations
  CONSTRAINT idx_unique_name UNIQUE(name)
);
