ALTER TABLE thread_messages
  ADD attachment BLOB NOT NULL DEFAULT '{}';

-- Migrate existing data to the new attachment column.
UPDATE thread_messages SET attachment = JSON_SET(attachment, '$.code', JSON(code_attachments)) WHERE code_attachments IS NOT NULL;
UPDATE thread_messages SET attachment = JSON_SET(attachment, '$.client_code', JSON(client_code_attachments)) WHERE client_code_attachments IS NOT NULL;
UPDATE thread_messages SET attachment = JSON_SET(attachment, '$.doc', JSON(doc_attachments)) WHERE doc_attachments IS NOT NULL;