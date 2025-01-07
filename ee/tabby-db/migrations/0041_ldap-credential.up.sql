CREATE TABLE ldap_credential(
  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,

  host STRING NOT NULL,
  port INTEGER NOT NULL DEFAULT 389,

  bind_dn STRING NOT NULL,
  bind_password STRING NOT NULL,
  base_dn STRING NOT NULL,
  user_filter STRING NOT NULL,

  -- enum of none, starttls, ldaps
  encryption STRING NOT NULL DEFAULT 'none',
  skip_tls_verify BOOLEAN NOT NULL DEFAULT FALSE,

  --- the attribute to be used as the Tabby user email address
  email_attribute STRING NOT NULL DEFAULT 'email',
  --- the attribute to be used as the Tabby user name
  name_attribute STRING,

  created_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now')),
  updated_at TIMESTAMP NOT NULL DEFAULT(DATETIME('now'))
);
