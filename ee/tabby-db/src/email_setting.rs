use anyhow::Result;
use rusqlite::{named_params, OptionalExtension};

use crate::DbConn;

const EMAIL_CREDENTIAL_ROW_ID: i32 = 1;

#[derive(Debug, PartialEq)]
pub struct EmailSettingDAO {
    pub smtp_username: String,
    pub smtp_password: String,
    pub smtp_server: String,
}

impl EmailSettingDAO {
    fn new(smtp_username: String, smtp_password: String, smtp_server: String) -> Self {
        Self {
            smtp_username,
            smtp_password,
            smtp_server,
        }
    }
}

impl DbConn {
    pub async fn read_email_setting(&self) -> Result<Option<EmailSettingDAO>> {
        let res = self
            .conn
            .call(|c| {
                Ok(c.query_row(
                    "SELECT smtp_username, smtp_password, smtp_server FROM email_setting WHERE id=?",
                    [EMAIL_CREDENTIAL_ROW_ID],
                    |row| Ok(EmailSettingDAO::new(row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional())
            })
            .await?;
        Ok(res?)
    }

    pub async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()> {
        Ok(self
            .conn
            .call(move |c| {
                let transaction = c.transaction()?;
                 let smtp_password = match smtp_password {
                    Some(pass) => pass,
                    None => {
                        transaction.query_row("SELECT smtp_password FROM email_setting WHERE id = ?", [EMAIL_CREDENTIAL_ROW_ID], |r| r.get(0))?
                    }
                };
                transaction.execute("INSERT INTO email_setting VALUES (:id, :user, :pass, :server)
                        ON CONFLICT(id) DO UPDATE SET smtp_username = :user, smtp_password = :pass, smtp_server = :server",
                        named_params! {
                            ":id": EMAIL_CREDENTIAL_ROW_ID,
                            ":user": smtp_username,
                            ":pass": smtp_password,
                            ":server": smtp_server,
                        }
                )?;
                transaction.commit()?;
                Ok(())
            })
            .await?)
    }

    pub async fn delete_email_setting(&self) -> Result<()> {
        Ok(self
            .conn
            .call(move |c| {
                c.execute(
                    "DELETE FROM email_setting WHERE id = ?",
                    [EMAIL_CREDENTIAL_ROW_ID],
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
        assert_eq!(conn.read_email_setting().await.unwrap(), None);

        // Test insertion
        conn.update_email_setting("user".into(), Some("pass".into()), "server".into())
            .await
            .unwrap();

        let creds = conn.read_email_setting().await.unwrap().unwrap();
        assert_eq!(creds.smtp_username, "user");
        assert_eq!(creds.smtp_password, "pass");
        assert_eq!(creds.smtp_server, "server");

        // Test update without password
        conn.update_email_setting("user2".into(), None, "server2".into())
            .await
            .unwrap();

        let creds = conn.read_email_setting().await.unwrap().unwrap();
        assert_eq!(creds.smtp_username, "user2");
        assert_eq!(creds.smtp_password, "pass");
        assert_eq!(creds.smtp_server, "server2");
    }
}
