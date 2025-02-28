ALTER TABLE pages ADD code_source_id VARCHAR(255);

ALTER TABLE page_sections ADD attachment BLOB NOT NULL DEFAULT '{}';
