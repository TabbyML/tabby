use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use lettre::{
    message::{Mailbox, MessageBuilder},
    transport::smtp::{
        authentication::{Credentials, Mechanism},
        client::{Tls, TlsParameters},
        AsyncSmtpTransportBuilder,
    },
    Address, AsyncSmtpTransport, AsyncTransport, Tokio1Executor,
};
use tabby_db::{DbConn, DbEnum};
use tokio::{sync::RwLock, task::JoinHandle};
use tracing::warn;

use crate::schema::{
    email::{AuthMethod, EmailService, EmailSetting, EmailSettingInput, Encryption},
    setting::SettingService,
};

pub struct EmailServiceImpl {
    db: DbConn,
    smtp_server: Arc<RwLock<Option<AsyncSmtpTransport<Tokio1Executor>>>>,
    from: RwLock<String>,
}

fn auth_mechanism(auth_method: AuthMethod) -> Vec<Mechanism> {
    match auth_method {
        AuthMethod::Plain => vec![Mechanism::Plain],
        AuthMethod::Login => vec![Mechanism::Login],
        AuthMethod::None => vec![],
    }
}

fn make_smtp_builder(
    host: &str,
    port: u16,
    encryption: Encryption,
) -> Result<AsyncSmtpTransportBuilder> {
    let tls_parameters = TlsParameters::new(host.into())?;

    let builder = match encryption {
        Encryption::StartTls => AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(host)
            .port(port)
            .tls(Tls::Required(tls_parameters)),
        Encryption::SslTls => AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(host)
            .port(port)
            .tls(Tls::Wrapper(tls_parameters)),
        Encryption::None => {
            AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(host).port(port)
        }
    };

    Ok(builder)
}

impl EmailServiceImpl {
    pub fn new(db: DbConn) -> Self {
        Self {
            db,
            smtp_server: Default::default(),
            from: Default::default(),
        }
    }

    /// (Re)initialize the SMTP server connection using new credentials
    async fn reset_smtp_connection(
        &self,
        username: String,
        password: String,
        host: &str,
        port: i32,
        from_address: &str,
        encryption: Encryption,
        auth_method: AuthMethod,
    ) -> Result<()> {
        let mut smtp_server = self.smtp_server.write().await;

        let mut builder = make_smtp_builder(host, port as u16, encryption)?;
        let mechanism = auth_mechanism(auth_method);
        if !mechanism.is_empty() {
            builder = builder
                .credentials(Credentials::new(username, password))
                .authentication(mechanism);
        }
        *smtp_server = Some(builder.build());
        *self.from.write().await = from_address.into();
        Ok(())
    }

    /// Close the SMTP server connection
    async fn shutdown_smtp_connection(&self) {
        *self.smtp_server.write().await = None;
    }

    async fn send_email_in_background(
        &self,
        to: String,
        subject: String,
        message: String,
    ) -> Result<JoinHandle<()>> {
        let smtp_server = self.smtp_server.clone();
        let from = self.from.read().await.clone();
        let address_from = to_address(from)?;
        let address_to = to_address(to)?;
        let msg = MessageBuilder::new()
            .subject(subject)
            .from(Mailbox::new(Some("Tabby Admin".to_owned()), address_from))
            .to(Mailbox::new(None, address_to))
            .body(message)
            .map_err(anyhow::Error::msg)?;

        if let Some(smtp_server) = &*(smtp_server.read().await) {
            // Not enabled.
            match smtp_server.send(msg).await.map_err(anyhow::Error::msg) {
                Ok(_) => {}
                Err(err) => {
                    warn!("Failed to send mail due to {}", err);
                }
            };
        }
        Ok(tokio::spawn(async move {}))
    }
}

pub async fn new_email_service(db: DbConn) -> Result<impl EmailService> {
    let setting = db.read_email_setting().await?;
    let service = EmailServiceImpl::new(db);

    // Optionally initialize the SMTP connection when the service is created
    if let Some(setting) = setting {
        let encryption = Encryption::from_enum_str(&setting.encryption)?;
        let auth_method = AuthMethod::from_enum_str(&setting.auth_method)?;
        service
            .reset_smtp_connection(
                setting.smtp_username,
                setting.smtp_password,
                &setting.smtp_server,
                setting.smtp_port as i32,
                &setting.from_address,
                encryption,
                auth_method,
            )
            .await?;
    };
    Ok(service)
}

#[async_trait]
impl EmailService for EmailServiceImpl {
    async fn read_email_setting(&self) -> Result<Option<EmailSetting>> {
        let setting = self.db.read_email_setting().await?;
        let Some(setting) = setting else {
            return Ok(None);
        };
        let setting = setting.try_into();
        let Ok(setting) = setting else {
            self.db.delete_email_setting().await?;
            warn!("Email settings are corrupt, and have been deleted. Please reset them.");
            return Ok(None);
        };
        Ok(Some(setting))
    }

    async fn update_email_setting(&self, input: EmailSettingInput) -> Result<()> {
        self.db
            .update_email_setting(
                input.smtp_username.clone(),
                input.smtp_password.clone(),
                input.smtp_server.clone(),
                input.smtp_port,
                input.from_address.clone(),
                input.encryption.as_enum_str().into(),
                input.auth_method.as_enum_str().into(),
            )
            .await?;
        let smtp_password = match input.smtp_password {
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
            input.smtp_username,
            smtp_password.clone(),
            &input.smtp_server,
            input.smtp_port,
            &input.from_address,
            input.encryption,
            input.auth_method,
        )
        .await?;
        Ok(())
    }

    async fn delete_email_setting(&self) -> Result<()> {
        self.db.delete_email_setting().await?;
        // When the SMTP credentials are deleted, close the SMTP server connection
        self.shutdown_smtp_connection().await;
        Ok(())
    }

    async fn send_invitation_email(&self, email: String, code: String) -> Result<JoinHandle<()>> {
        let network_setting = self.db.read_network_setting().await?;
        let external_url = network_setting.external_url;
        self.send_email_in_background(
            email,
            "You've been invited to join a Tabby workspace!".into(),
            format!("Welcome to Tabby! You have been invited to join a Tabby Server, where you can tap into AI-driven code completions and chat assistants.\n\nGo to {external_url}/auth/signup?invitationCode={code} to join!"),
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
    use std::time::Duration;

    use serde::Deserialize;
    use tokio::process::{Child, Command};

    use super::*;

    #[tokio::test]
    async fn test_update_email_with_service() {
        let db: DbConn = DbConn::new_in_memory().await.unwrap();
        let service = EmailServiceImpl::new(db);

        let update_input = EmailSettingInput {
            smtp_username: "test@example.com".into(),
            from_address: "test".into(),
            smtp_server: "smtp://example.com".into(),
            smtp_port: 578,
            encryption: Encryption::SslTls,
            auth_method: AuthMethod::None,
            smtp_password: Some("123456".to_owned()),
        };
        service.update_email_setting(update_input).await.unwrap();
        let setting = service.read_email_setting().await.unwrap().unwrap();
        assert_eq!(setting.smtp_username, "test@example.com");

        service.delete_email_setting().await.unwrap();
    }

    fn default_email_setting() -> EmailSetting {
        EmailSetting {
            smtp_username: "tabby".into(),
            smtp_server: "127.0.0.1".into(),
            smtp_port: 1025,
            from_address: "tabby@localhost".into(),
            encryption: Encryption::None,
            auth_method: AuthMethod::None,
        }
    }

    async fn start_smtp_server() -> Child {
        let mut cmd = Command::new("mailtutan");
        cmd.kill_on_drop(true);

        let child = cmd
            .spawn()
            .expect("You need to run `cargo install mailtutan` before running this test");
        tokio::time::sleep(Duration::from_secs(1)).await;
        child
    }

    #[derive(Deserialize)]
    struct Mail {
        sender: String,
    }

    async fn read_mails() -> Vec<Mail> {
        reqwest::get("http://localhost:1080/api/messages")
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    /*
     * Requires https://github.com/mailtutan/mailtutan
     */
    #[tokio::test]
    async fn test_send_email() {
        let email_setting = default_email_setting();
        let smtp_password = "fake";
        let mut child = start_smtp_server().await;

        let db: DbConn = DbConn::new_in_memory().await.unwrap();
        db.update_email_setting(
            email_setting.smtp_username,
            Some(smtp_password.into()),
            email_setting.smtp_server,
            email_setting.smtp_port,
            email_setting.from_address.clone(),
            email_setting.encryption.as_enum_str().into(),
            email_setting.auth_method.as_enum_str().into(),
        )
        .await
        .unwrap();

        let service = new_email_service(db).await.unwrap();

        let handle = service
            .send_invitation_email("user@localhost".into(), "12345".into())
            .await
            .unwrap();

        handle.await.unwrap();

        let mails = read_mails().await;
        assert!(mails[0].sender.contains(&email_setting.from_address));
        child.kill().await.unwrap();
    }
}
