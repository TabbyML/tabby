-- Add up migration script here
ALTER TABLE web_crawler_urls ADD COLUMN web_name VARCHAR(255) NOT NULL DEFAULT '';
ALTER TABLE web_crawler_urls ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE web_crawler_urls ADD COLUMN is_preset BOOLEAN NOT NULL DEFAULT FALSE;
