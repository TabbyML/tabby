CREATE TABLE IF NOT EXISTS email_setting(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smtp_username VARCHAR(255) NOT NULL,
    smtp_password VARCHAR(255) NOT NULL,
    smtp_server VARCHAR(255) NOT NULL
);
