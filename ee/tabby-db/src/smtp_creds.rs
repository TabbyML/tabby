use anyhow::Result;
use rusqlite::OptionalExtension;

use crate::DbConn;

pub struct SMTPInfo {
    pub email: String,
    pub password: String,
    pub mailserver_url: String,
}

impl SMTPInfo {
    fn new(email: String, password: String, mailserver_url: String) -> Self {
        Self {
            email,
            password,
            mailserver_url,
        }
    }
}

impl DbConn {
    pub async fn get_smtp_info(&self) -> Result<Option<SMTPInfo>> {
        let res = self
            .conn
            .call(|c| {
                Ok(c.query_row(
                    "SELECT email, password, mailserver_url FROM smtp_info",
                    (),
                    |row| Ok(SMTPInfo::new(row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional())
            })
            .await?;
        // Unsure why the map_err is needed. The `?` from the previous line
        // should convert it automatically, but this breaks without it.
        res.map_err(Into::into)
    }

    pub async fn update_smtp_info(&self, creds: SMTPInfo) -> Result<()> {
        Ok(self
            .conn
            .call(move |c| {
                c.execute("DELETE FROM smtp_info", ())?;
                c.execute(
                    "INSERT INTO smtp_info VALUES (?, ?, ?)",
                    (creds.email, creds.password, creds.mailserver_url),
                )?;
                Ok(())
            })
            .await?)
    }
}
