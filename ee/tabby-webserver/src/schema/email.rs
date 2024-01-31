use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;

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
    async fn send_mail(&self, to: String, subject: String, body: String) -> Result<()>;
}
