use anyhow::Result;
use rusqlite::OptionalExtension;

use crate::DbConn;

const MAIL_CREDENTIAL_ROW_ID: i32 = 1;

#[derive(Debug, PartialEq)]
pub struct EmailServiceCredentialDAO {
    pub smtp_username: String,
    pub smtp_password: String,
    pub smtp_server: String,
}

impl EmailServiceCredentialDAO {
    fn new(email: String, password: String, mailserver_url: String) -> Self {
        Self {
            smtp_username: email,
            smtp_password: password,
            smtp_server: mailserver_url,
        }
    }
}

impl DbConn {
    pub async fn get_email_service_credential(&self) -> Result<Option<EmailServiceCredentialDAO>> {
        let res = self
            .conn
            .call(|c| {
                Ok(c.query_row(
                    "SELECT smtp_username, smtp_password, smtp_server FROM email_service_credential WHERE id=?",
                    [MAIL_CREDENTIAL_ROW_ID],
                    |row| Ok(EmailServiceCredentialDAO::new(row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional())
            })
            .await?;
        // Unsure why the map_err is needed. The `?` from the previous line
        // should convert it automatically, but this breaks without it.
        res.map_err(Into::into)
    }

    pub async fn update_email_service_credential(
        &self,
        creds: EmailServiceCredentialDAO,
    ) -> Result<()> {
        Ok(self
            .conn
            .call(move |c| {
                c.execute("DELETE FROM email_service_credential", ())?;
                c.execute(
                    "INSERT INTO email_service_credential VALUES (?, ?, ?, ?)",
                    (
                        MAIL_CREDENTIAL_ROW_ID,
                        creds.smtp_username,
                        creds.smtp_password,
                        creds.smtp_server,
                    ),
                )?;
                Ok(())
            })
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_smtp_info() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // Test no credentials prior to insertion
        assert_eq!(conn.get_email_service_credential().await.unwrap(), None);

        // Test insertion
        conn.update_email_service_credential(EmailServiceCredentialDAO::new(
            "user".into(),
            "pass".into(),
            "server".into(),
        ))
        .await
        .unwrap();

        let creds = conn.get_email_service_credential().await.unwrap();
        assert_eq!(creds.smtp_username, "user");
        assert_eq!(creds.smtp_password, "pass");
        assert_eq!(creds.smtp_server, "server");
    }
}
