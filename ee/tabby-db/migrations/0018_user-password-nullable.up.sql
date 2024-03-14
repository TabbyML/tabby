ALTER TABLE users ADD COLUMN password_encrypted_nullable VARCHAR(128);

UPDATE users SET password_encrypted_nullable =
    CASE
        WHEN LENGTH(password_encrypted) = 0 THEN NULL
        ELSE password_encrypted
    END;

ALTER TABLE users DROP COLUMN password_encrypted;
ALTER TABLE users RENAME password_encrypted_nullable TO password_encrypted;
