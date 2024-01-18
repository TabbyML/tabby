use anyhow::{anyhow, Result};
use async_trait::async_trait;
use lettre::{
    transport::smtp::{authentication::Credentials},
    Message, SmtpTransport, Transport,
};
use tabby_db::DbConn;

use crate::schema::email::{EmailService, EmailSetting};

#[async_trait]
impl EmailService for DbConn {
    async fn get_email_setting(&self) -> Result<Option<EmailSetting>> {
        let creds = self.read_email_setting().await?;
        Ok(creds.map(Into::into))
    }

    async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()> {
        self.update_email_setting(smtp_username, smtp_password, smtp_server)
            .await
    }

    async fn delete_email_setting(&self) -> Result<()> {
        self.delete_email_setting().await
    }

    async fn send_mail(&self, messages: &mut Vec<Message>) -> Result<()> {
        let settings = self
            .read_email_setting()
            .await?
            .ok_or_else(|| anyhow!("email settings not specified, cannot send mail"))?;
        let server = SmtpTransport::relay(&settings.smtp_server)?
            .credentials(Credentials::new(
                settings.smtp_username,
                settings.smtp_password,
            ))
            .build();
        while let Some(next) = messages.last() {
            server.send(next)?;
            messages.remove(messages.len() - 1);
        }
        Ok(())
    }
}
