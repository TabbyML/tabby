use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject};
use tokio::task::JoinHandle;
use validator::Validate;

use crate::schema::Result;

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

#[derive(GraphQLObject)]
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
    #[validate(email(code = "fromAddress", message = "Invalid email address"))]
    pub from_address: String,
    pub smtp_server: String,
    #[validate(range(min = 1, max = 65535, code = "smtpPort", message = "Invalid port"))]
    pub smtp_port: i32,
    pub encryption: Encryption,
    pub auth_method: AuthMethod,
    pub smtp_password: Option<String>,
}

#[async_trait]
pub trait EmailService: Send + Sync {
    async fn read_setting(&self) -> Result<Option<EmailSetting>>;
    async fn update_setting(&self, input: EmailSettingInput) -> Result<()>;
    async fn delete_setting(&self) -> Result<()>;

    async fn send_test(&self, to: String) -> Result<JoinHandle<()>>;
    async fn send_password_reset(&self, to: String, code: String) -> Result<JoinHandle<()>>;
    async fn send_invitation(&self, email: String, code: String) -> Result<JoinHandle<()>>;
    async fn send_signup(&self, email: String) -> Result<JoinHandle<()>>;
}
