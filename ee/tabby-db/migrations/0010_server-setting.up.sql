CREATE TABLE server_setting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    security_disable_clientside_telemetry BOOLEAN NOT NULL DEFAULT FALSE,
    network_external_url STRING NOT NULL DEFAULT 'http://localhost:8080',
    security_allowed_register_domain_list STRING
);
