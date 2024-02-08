use anyhow::{anyhow, Result};
use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLObject};
use serde::{Deserialize, Serialize};
use validator::validate_url;

#[derive(GraphQLEnum, Clone, Debug)]
pub enum Encryption {
    StartTls,
    SslTls,
    None,
}

impl ToString for Encryption {
    fn to_string(&self) -> String {
        match self {
            Encryption::StartTls => "STARTTLS",
            Encryption::SslTls => "SSLTLS",
            Encryption::None => "NONE",
        }
        .into()
    }
}

#[derive(GraphQLEnum, Clone, Debug)]
pub enum AuthMethod {
    Plain,
    Login,
    XOAuth2,
}

impl ToString for AuthMethod {
    fn to_string(&self) -> String {
        match self {
            AuthMethod::Plain => "PLAIN",
            AuthMethod::Login => "LOGIN",
            AuthMethod::XOAuth2 => "XOAUTH2",
        }
        .into()
    }
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

        let encryption = Self::convert_encryption(&encryption)?;
        let auth_method = Self::convert_auth_method(&auth_method)?;

        let from_address = from_address.unwrap_or_else(|| smtp_username.clone());
        Ok(EmailSetting {
            smtp_username,
            smtp_server,
            from_address,
            encryption,
            auth_method,
        })
    }

    pub fn convert_encryption(encryption: &str) -> Result<Encryption> {
        Ok(match &*encryption.to_lowercase() {
            "starttls" => Encryption::StartTls,
            "ssltls" | "ssl/tls" => Encryption::SslTls,
            "none" => Encryption::None,
            _ => return Err(anyhow!("Invalid encryption setting")),
        })
    }

    pub fn convert_auth_method(auth_method: &str) -> Result<AuthMethod> {
        Ok(match &*auth_method.to_lowercase() {
            "plain" => AuthMethod::Plain,
            "login" => AuthMethod::Login,
            "xoauth2" => AuthMethod::XOAuth2,
            _ => return Err(anyhow!("Invalid authentication method")),
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
