use anyhow::Result;
use sqlx::{query, query_as, query_scalar};

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
        let setting = query_as!(
            EmailSettingDAO,
            "SELECT smtp_username, smtp_password, smtp_server FROM email_setting WHERE id=?",
            EMAIL_CREDENTIAL_ROW_ID
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(setting)
    }

    pub async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()> {
        let mut transaction = self.pool.begin().await?;
        let smtp_password = match smtp_password {
            Some(pass) => pass,
            None => {
                query_scalar!(
                    "SELECT smtp_password FROM email_setting WHERE id = ?",
                    EMAIL_CREDENTIAL_ROW_ID
                )
                .fetch_one(&mut *transaction)
                .await?
            }
        };
        query!("INSERT INTO email_setting VALUES (:id, :user, :pass, :server)
                ON CONFLICT(id) DO UPDATE SET smtp_username = :user, smtp_password = :pass, smtp_server = :server",
            EMAIL_CREDENTIAL_ROW_ID,
            smtp_username,
            smtp_password,
            smtp_server).execute(&mut *transaction).await?;
        transaction.commit().await?;
        Ok(())
    }

    pub async fn delete_email_setting(&self) -> Result<()> {
        query!(
            "DELETE FROM email_setting WHERE id = ?",
            EMAIL_CREDENTIAL_ROW_ID
        )
        .execute(&self.pool)
        .await?;
        Ok(())
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
