DROP TABLE email_setting;
CREATE TABLE IF NOT EXISTS email_setting(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smtp_username VARCHAR(255) NOT NULL,
    smtp_password VARCHAR(255) NOT NULL,
    smtp_server VARCHAR(255) NOT NULL,
    from_address VARCHAR(255) NOT NULL,
    encryption VARCHAR(255) NOT NULL DEFAULT 'ssltls',
    auth_method VARCHAR(255) NOT NULL DEFAULT 'plain'
);
