use anyhow::{anyhow, Result};
use sqlx::{query, query_as, query_scalar};

use crate::DbConn;

const EMAIL_CREDENTIAL_ROW_ID: i32 = 1;

#[derive(Debug, PartialEq)]
pub struct EmailSettingDAO {
    pub smtp_username: String,
    pub smtp_password: String,
    pub smtp_server: String,
    pub smtp_port: i64,
    pub from_address: String,
    pub encryption: String,
    pub auth_method: String,
}

impl DbConn {
    pub async fn read_email_setting(&self) -> Result<Option<EmailSettingDAO>> {
        let setting = query_as!(
            EmailSettingDAO,
            "SELECT smtp_username, smtp_password, smtp_server, smtp_port, from_address, encryption, auth_method FROM email_setting WHERE id=?",
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
        smtp_port: i32,
        from_address: String,
        encryption: String,
        auth_method: String,
    ) -> Result<()> {
        let mut transaction = self.pool.begin().await?;
        let smtp_password = match smtp_password {
            Some(pass) => pass,
            None => query_scalar!(
                "SELECT smtp_password FROM email_setting WHERE id = ?",
                EMAIL_CREDENTIAL_ROW_ID
            )
            .fetch_one(&mut *transaction)
            .await
            .map_err(|_| anyhow!("smtp_password is required to enable email sending"))?,
        };
        query!("INSERT INTO email_setting (id, smtp_username, smtp_password, smtp_server, from_address, encryption, auth_method, smtp_port) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT(id) DO UPDATE SET smtp_username = $2, smtp_password = $3, smtp_server = $4, from_address = $5, encryption = $6, auth_method = $7, smtp_port = $8",
            EMAIL_CREDENTIAL_ROW_ID,
            smtp_username,
            smtp_password,
            smtp_server,
            from_address,
            encryption,
            auth_method,
            smtp_port,
        ).execute(&mut *transaction).await?;
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
        conn.update_email_setting(
            "user".into(),
            Some("pass".into()),
            "server".into(),
            25,
            "user".into(),
            "STARTTLS".into(),
            "".into(),
        )
        .await
        .unwrap();

        let creds = conn.read_email_setting().await.unwrap().unwrap();
        assert_eq!(creds.smtp_username, "user");
        assert_eq!(creds.smtp_password, "pass");
        assert_eq!(creds.smtp_server, "server");

        // Test update without password

        conn.update_email_setting(
            "user2".into(),
            None,
            "server2".into(),
            25,
            "user2".into(),
            "STARTTLS".into(),
            "".into(),
        )
        .await
        .unwrap();

        let creds = conn.read_email_setting().await.unwrap().unwrap();
        assert_eq!(creds.smtp_username, "user2");
        assert_eq!(creds.smtp_password, "pass");
        assert_eq!(creds.smtp_server, "server2");
    }
}
