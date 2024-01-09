use anyhow::Result;
use rusqlite::OptionalExtension;

use crate::DbConn;

const EMAIL_CREDENTIAL_ROW_ID: i32 = 1;

#[derive(Debug, PartialEq)]
pub struct EmailServiceCredentialDAO {
    pub smtp_username: String,
    pub smtp_password: String,
    pub smtp_server: String,
}

impl EmailServiceCredentialDAO {
    fn new(smtp_username: String, smtp_password: String, smtp_server: String) -> Self {
        Self {
            smtp_username,
            smtp_password,
            smtp_server,
        }
    }
}

impl DbConn {
    pub async fn read_email_service_credential(&self) -> Result<Option<EmailServiceCredentialDAO>> {
        let res = self
            .conn
            .call(|c| {
                Ok(c.query_row(
                    "SELECT smtp_username, smtp_password, smtp_server FROM email_service_credential WHERE id=?",
                    [EMAIL_CREDENTIAL_ROW_ID],
                    |row| Ok(EmailServiceCredentialDAO::new(row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional())
            })
            .await?;
        Ok(res?)
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
                        EMAIL_CREDENTIAL_ROW_ID,
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
        assert_eq!(conn.read_email_service_credential().await.unwrap(), None);

        // Test insertion
        conn.update_email_service_credential(EmailServiceCredentialDAO::new(
            "user".into(),
            "pass".into(),
            "server".into(),
        ))
        .await
        .unwrap();

        let creds = conn.read_email_service_credential().await.unwrap().unwrap();
        assert_eq!(creds.smtp_username, "user");
        assert_eq!(creds.smtp_password, "pass");
        assert_eq!(creds.smtp_server, "server");
    }
}
