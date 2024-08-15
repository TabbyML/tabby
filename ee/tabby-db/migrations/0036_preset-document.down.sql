-- Add down migration script here

ALTER TABLE web_crawler_urls DROP COLUMN name;
ALTER TABLE web_crawler_urls DROP COLUMN active;
ALTER TABLE web_crawler_urls DROP COLUMN is_preset;
