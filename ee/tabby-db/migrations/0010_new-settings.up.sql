ALTER TABLE email_setting ADD COLUMN from_address VARCHAR(255);
ALTER TABLE email_setting ADD COLUMN encryption VARCHAR(255) NOT NULL DEFAULT 'STARTTLS';
ALTER TABLE email_setting ADD COLUMN auth_method VARCHAR(255) NOT NULL DEFAULT 'PLAIN';

CREATE TABLE server_setting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disable_clientside_telemetry BOOLEAN NOT NULL DEFAULT FALSE,
    external_url STRING NOT NULL DEFAULT 'http://localhost:8080',
    allowed_register_domains STRING
);
