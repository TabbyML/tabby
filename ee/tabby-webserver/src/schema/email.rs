use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;
use lettre::Message;

#[derive(GraphQLObject)]
pub struct EmailSetting {
    pub smtp_username: String,
    pub smtp_server: String,
}

#[async_trait]
pub trait EmailService: Send + Sync {
    async fn get_email_setting(&self) -> Result<Option<EmailSetting>>;
    async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()>;
    async fn delete_email_setting(&self) -> Result<()>;
    /// Send emails using the saved email settings. Any emails that don't get sent due to errors will remain in the Vec.
    async fn send_mail(&self, messages: &Message) -> Result<()>;
}
