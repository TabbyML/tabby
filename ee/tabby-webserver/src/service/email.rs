use anyhow::Result;
use async_trait::async_trait;
use tabby_db::DbConn;

use crate::schema::email::{EmailService, EmailSettings};

#[async_trait]
impl EmailService for DbConn {
    async fn get_email_settings(&self) -> Result<Option<EmailSettings>> {
        let creds = self.read_email_settings().await?;
        Ok(creds.map(Into::into))
    }

    async fn update_email_settings(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()> {
        self.update_email_settings(smtp_username, smtp_password, smtp_server)
            .await
    }

    async fn delete_email_settings(&self) -> Result<()> {
        self.delete_email_settings().await
    }
}
