use anyhow::Result;
use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject};
use thiserror::Error;
use tokio::task::JoinHandle;
use validator::Validate;

#[derive(GraphQLEnum, Clone, Debug)]
pub enum Encryption {
    StartTls,
    SslTls,
    None,
}

#[derive(GraphQLEnum, Clone, Debug)]
pub enum AuthMethod {
    None,
    Plain,
    Login,
}

#[derive(GraphQLObject, Clone)]
pub struct EmailSetting {
    pub smtp_username: String,
    pub smtp_server: String,
    pub smtp_port: i32,
    pub from_address: String,
    pub encryption: Encryption,
    pub auth_method: AuthMethod,
}

#[derive(GraphQLInputObject, Validate)]
pub struct EmailSettingInput {
    pub smtp_username: String,
    #[validate(email(code = "fromAddress"))]
    pub from_address: String,
    pub smtp_server: String,
    pub smtp_port: i32,
    pub encryption: Encryption,
    pub auth_method: AuthMethod,
    pub smtp_password: Option<String>,
}

#[derive(Error, Debug)]
pub enum SendEmailError {
    #[error("Email service is not configured")]
    NotConfigured,

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[async_trait]
pub trait EmailService: Send + Sync {
    async fn read_email_setting(&self) -> Result<Option<EmailSetting>>;
    async fn update_email_setting(&self, input: EmailSettingInput) -> Result<()>;
    async fn delete_email_setting(&self) -> Result<()>;

    async fn send_invitation_email(
        &self,
        email: String,
        code: String,
    ) -> Result<JoinHandle<()>, SendEmailError>;
}
