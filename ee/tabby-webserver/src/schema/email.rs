use anyhow::Result;
use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLObject};
use validator::Validate;

#[derive(GraphQLEnum, Clone, Debug)]
pub enum Encryption {
    StartTls,
    SslTls,
    None,
}

#[derive(GraphQLEnum, Clone, Debug)]
pub enum AuthMethod {
    Plain,
    Login,
    XOAuth2,
}

#[derive(GraphQLObject)]
pub struct EmailSetting {
    pub smtp_username: String,
    pub smtp_server: String,
    pub from_address: String,
    pub encryption: Encryption,
    pub auth_method: AuthMethod,
}

#[derive(GraphQLObject, Validate)]
pub struct EmailSettingInput {
    #[validate(email)]
    pub smtp_username: String,
    #[validate(email)]
    pub from_address: String,
    #[validate(url)]
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
        from_address: String,
        encryption: Encryption,
        auth_method: AuthMethod,
    ) -> Result<()>;
    async fn delete_email_setting(&self) -> Result<()>;

    async fn send_invitation_email(&self, email: String, code: String) -> Result<()>;
}
