use anyhow::Result;
use rusqlite::OptionalExtension;

use crate::DbConn;

const MAIL_CREDENTIAL_ROW_ID: i32 = 1;

pub struct SMTPInfoDAO {
    pub email: String,
    pub password: String,
    pub mailserver_url: String,
}

impl SMTPInfoDAO {
    fn new(email: String, password: String, mailserver_url: String) -> Self {
        Self {
            email,
            password,
            mailserver_url,
        }
    }
}

impl DbConn {
    pub async fn get_smtp_info(&self) -> Result<Option<SMTPInfoDAO>> {
        let res = self
            .conn
            .call(|c| {
                Ok(c.query_row(
                    "SELECT smtp_username, smtp_password, smtp_server FROM email_service_credentials WHERE id=?",
                    [MAIL_CREDENTIAL_ROW_ID],
                    |row| Ok(SMTPInfoDAO::new(row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional())
            })
            .await?;
        // Unsure why the map_err is needed. The `?` from the previous line
        // should convert it automatically, but this breaks without it.
        res.map_err(Into::into)
    }

    pub async fn update_smtp_info(&self, creds: SMTPInfoDAO) -> Result<()> {
        Ok(self
            .conn
            .call(move |c| {
                c.execute("DELETE FROM smtp_info", ())?;
                c.execute(
                    "INSERT INTO smtp_info VALUES (?, ?, ?, ?)",
                    (
                        MAIL_CREDENTIAL_ROW_ID,
                        creds.email,
                        creds.password,
                        creds.mailserver_url,
                    ),
                )?;
                Ok(())
            })
            .await?)
    }
}
