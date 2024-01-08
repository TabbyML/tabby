CREATE TABLE email_service_credentials(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smtp_username VARCHAR(255),
    smtp_password VARCHAR(255),
    smtp_server VARCHAR(255),
);
