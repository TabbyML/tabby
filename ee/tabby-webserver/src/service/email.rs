use anyhow::{anyhow, Result};
use async_trait::async_trait;
use lettre::{
    message::{Mailbox, MessageBuilder},
    transport::smtp::{
        authentication::{Credentials, Mechanism},
        SmtpTransportBuilder,
    },
    Address, SmtpTransport, Transport,
};
use tabby_db::{DbConn, DbEnum};
use tokio::sync::RwLock;
use tracing::warn;

use crate::schema::email::{AuthMethod, EmailService, EmailSetting, Encryption};

struct EmailServiceImpl {
    db: DbConn,
    smtp_server: RwLock<Option<SmtpTransport>>,
    from: RwLock<String>,
}

fn auth_mechanism(auth_method: AuthMethod) -> Mechanism {
    match auth_method {
        AuthMethod::Plain => Mechanism::Plain,
        AuthMethod::Login => Mechanism::Login,
        AuthMethod::XOAuth2 => Mechanism::Xoauth2,
    }
}

fn make_smtp_builder(address: &str, encryption: Encryption) -> Result<SmtpTransportBuilder> {
    match encryption {
        Encryption::StartTls => Ok(SmtpTransport::starttls_relay(address)?),
        Encryption::SslTls => Ok(SmtpTransport::relay(address)?),
        Encryption::None => Ok(SmtpTransport::builder_dangerous(address)),
    }
}

impl EmailServiceImpl {
    /// (Re)initialize the SMTP server connection using new credentials
    async fn reset_smtp_connection(
        &self,
        username: String,
        password: String,
        server: &str,
        encryption: Encryption,
        auth_method: AuthMethod,
    ) -> Result<()> {
        let mut smtp_server = self.smtp_server.write().await;
        *smtp_server = Some(
            make_smtp_builder(server, encryption)?
                .credentials(Credentials::new(username, password))
                .authentication(vec![auth_mechanism(auth_method)])
                .build(),
        );
        Ok(())
    }

    /// Close the SMTP server connection
    async fn shutdown_smtp_connection(&self) {
        *self.smtp_server.write().await = None;
    }

    async fn send_mail(&self, to: String, subject: String, message: String) -> Result<()> {
        let smtp_server = self.smtp_server.read().await;
        let Some(smtp_server) = &*smtp_server else {
            return Err(anyhow!("email settings have not been populated"));
        };
        let from = self.from.read().await;
        let address_from = to_address(from.clone())?;
        let address_to = to_address(to)?;
        let msg = MessageBuilder::new()
            .subject(subject)
            .from(Mailbox::new(Some("Tabby".into()), address_from))
            .to(Mailbox::new(None, address_to))
            .body(message)?;
        smtp_server.send(&msg)?;
        Ok(())
    }
}

pub async fn new_email_service(db: DbConn) -> Result<impl EmailService> {
    let creds = db.read_email_setting().await?;
    let service = EmailServiceImpl {
        db,
        smtp_server: Default::default(),
        from: Default::default(),
    };
    // Optionally initialize the SMTP connection when the service is created
    if let Some(creds) = creds {
        let encryption = Encryption::from_enum_str(&creds.encryption)?;
        let auth_method = AuthMethod::from_enum_str(&creds.auth_method)?;
        service
            .reset_smtp_connection(
                creds.smtp_username,
                creds.smtp_password,
                &creds.smtp_server,
                encryption,
                auth_method,
            )
            .await?;
    };
    Ok(service)
}

#[async_trait]
impl EmailService for EmailServiceImpl {
    async fn get_email_setting(&self) -> Result<Option<EmailSetting>> {
        let creds = self.db.read_email_setting().await?;
        let Some(creds) = creds else {
            return Ok(None);
        };
        let creds = creds.try_into();
        let Ok(creds) = creds else {
            self.db.delete_email_setting().await?;
            warn!("Email settings are corrupt, and have been deleted. Please reset them.");
            return Ok(None);
        };
        Ok(Some(creds))
    }

    async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
        from_address: String,
        encryption: Encryption,
        auth_method: AuthMethod,
    ) -> Result<()> {
        self.db
            .update_email_setting(
                smtp_username.clone(),
                smtp_password.clone(),
                smtp_server.clone(),
                from_address.clone(),
                encryption.as_enum_str().into(),
                auth_method.as_enum_str().into(),
            )
            .await?;
        *self.from.write().await = smtp_username.clone();
        let smtp_password = match smtp_password {
            Some(pass) => pass,
            None => {
                self.db
                    .read_email_setting()
                    .await?
                    .expect("error already happened if entry is missing and password is empty")
                    .smtp_password
            }
        };
        // When the SMTP credentials are updated, reinitialize the SMTP server connection
        self.reset_smtp_connection(
            smtp_username,
            smtp_password.clone(),
            &smtp_server,
            encryption,
            auth_method,
        )
        .await?;
        Ok(())
    }

    async fn delete_email_setting(&self) -> Result<()> {
        self.delete_email_setting().await?;
        // When the SMTP credentials are deleted, close the SMTP server connection
        self.shutdown_smtp_connection().await;
        Ok(())
    }

    async fn send_invitation_email(&self, email: String, code: String) -> Result<()> {
        // TODO: Include invitation link
        self.send_mail(
            email,
            "You've been invited to join a Tabby workspace!".into(),
            format!("Welcome to Tabby! You have been invited to join a Tabby instance, where you can tap into\
                AI-driven code completions and chat assistants. Your invite code is {code}, use this at the link\
                you were given to join the organization."),
        ).await
    }
}

fn to_address(email: String) -> Result<Address> {
    let (user, domain) = email
        .split_once('@')
        .ok_or_else(|| anyhow!("address contains no @"))?;
    Ok(Address::new(user, domain)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_update_email_with_service() {
        let db: DbConn = DbConn::new_in_memory().await.unwrap();
        let service = EmailServiceImpl {
            db,
            smtp_server: Default::default(),
            from: Default::default(),
        };
        service
            .update_email_setting(
                "test@example.com".into(),
                Some("test".into()),
                "smtp://example.com".into(),
                "test@example.com".into(),
                Encryption::SslTls,
                AuthMethod::Plain,
            )
            .await
            .unwrap();
        let setting = service.get_email_setting().await.unwrap().unwrap();
        assert_eq!(setting.smtp_username, "test@example.com");
    }
}
