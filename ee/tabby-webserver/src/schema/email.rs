use anyhow::{anyhow, Result};
use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLObject};
use tabby_db::conversions::DatabaseSerializable;
use validator::validate_url;

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

impl EmailSetting {
    pub fn new_validate(
        smtp_username: String,
        smtp_server: String,
        from_address: Option<String>,
        encryption: String,
        auth_method: String,
    ) -> Result<Self> {
        if !validate_url(&smtp_server) {
            return Err(anyhow!("Invalid smtp server address"));
        }

        let encryption = Encryption::from_db_str(&encryption)?;
        let auth_method = AuthMethod::from_db_str(&auth_method)?;

        let from_address = from_address.unwrap_or_else(|| smtp_username.clone());
        Ok(EmailSetting {
            smtp_username,
            smtp_server,
            from_address,
            encryption,
            auth_method,
        })
    }
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
        encryption: Encryption,
        auth_method: AuthMethod,
    ) -> Result<()>;
    async fn delete_email_setting(&self) -> Result<()>;

    async fn send_invitation_email(&self, email: String, code: String) -> Result<()>;
}
