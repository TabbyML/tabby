UPDATE users SET password_encrypted =
    CASE
        WHEN password_encrypted IS NULL THEN ''
        ELSE password_encrypted
    END;
