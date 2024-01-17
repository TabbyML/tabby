use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;

#[derive(GraphQLObject)]
pub struct EmailSettings {
    pub smtp_username: String,
    pub smtp_server: String,
}

#[async_trait]
pub trait EmailService: Send + Sync {
    async fn get_email_settings(&self) -> Result<Option<EmailSettings>>;
    async fn update_email_settings(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()>;
    async fn delete_email_settings(&self) -> Result<()>;
}
