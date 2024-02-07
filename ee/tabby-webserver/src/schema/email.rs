use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;
use validator::Validate;

#[derive(GraphQLObject, Validate)]
pub struct EmailSetting {
    pub smtp_username: String,
    #[validate(url)]
    pub smtp_server: String,
    pub from_address: String,
    pub encryption: String,
    pub auth_method: String,
}

#[async_trait]
pub trait EmailService: Send + Sync {
    async fn get_email_setting(&self) -> Result<Option<EmailSetting>>;
    async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
        from_address: Option<String>,
        encryption: String,
        auth_method: String,
    ) -> Result<()>;
    async fn delete_email_setting(&self) -> Result<()>;

    async fn send_invitation_email(&self, email: String, code: String) -> Result<()>;
}
