use anyhow::Result;
use async_trait::async_trait;
use tabby_db::DbConn;

use crate::schema::email::{EmailService, EmailServiceCredential};

#[async_trait]
impl EmailService for DbConn {
    async fn get_email_service_credential(&self) -> Result<Option<EmailServiceCredential>> {
        let creds = self.read_email_service_credential().await?;
        Ok(creds.map(Into::into))
    }

    async fn update_email_service_credential(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()> {
        self.update_email_service_credential(smtp_username, smtp_password, smtp_server)
            .await
    }

    async fn delete_email_service_credential(&self) -> Result<()> {
        self.delete_email_service_credential().await
    }
}
