use anyhow::Result;
use async_trait::async_trait;
use tabby_db::{DbConn, EmailServiceCredentialDAO};

use crate::schema::email_service_credential::{
    EmailServiceCredential, EmailServiceCredentialService,
};

#[async_trait]
impl EmailServiceCredentialService for DbConn {
    async fn get_email_service_credential(&self) -> Result<Option<EmailServiceCredential>> {
        let creds = self.read_email_service_credential().await?;
        Ok(creds.map(Into::into))
    }

    async fn update_email_service_credential(
        &self,
        creds: EmailServiceCredentialDAO,
    ) -> Result<()> {
        self.update_email_service_credential(EmailServiceCredentialDAO {
            smtp_username: creds.smtp_username,
            smtp_password: creds.smtp_password,
            smtp_server: creds.smtp_server,
        })
        .await
    }
}
