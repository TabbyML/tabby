use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use lettre::{
    message::{header::ContentType, Mailbox, MessageBuilder},
    transport::smtp::{
        authentication::{Credentials, Mechanism},
        client::{Certificate, Tls, TlsParameters},
        AsyncSmtpTransportBuilder,
    },
    Address, AsyncSmtpTransport, AsyncTransport, Tokio1Executor,
};
use tabby_db::DbConn;
use tokio::{sync::RwLock, task::JoinHandle};
use tracing::warn;
mod templates;
#[cfg(test)]
pub mod testutils;

use tabby_schema::{
    email::{AuthMethod, EmailService, EmailSetting, EmailSettingInput, Encryption},
    setting::SettingService,
    CoreError, DbEnum, Result,
};

use super::setting;

struct EmailServiceImpl {
    db: DbConn,
    setting: Arc<dyn SettingService>,
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
    _cert_pem: Option<String>,
) -> Result<AsyncSmtpTransportBuilder> {
    let mut tls_parameters = TlsParameters::builder(host.into());

    if let Ok(cert_string) = std::env::var("TABBY_WEBSERVER_EMAIL_CERT") {
        let cert = Certificate::from_pem(cert_string.as_bytes())
            .map_err(|e| CoreError::Other(e.into()))?;
        tls_parameters = tls_parameters.add_root_certificate(cert);
    }

    let tls_parameters = tls_parameters
        .build()
        .map_err(|e| CoreError::Other(e.into()))?;

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
    fn new(db: DbConn, setting: Arc<dyn SettingService>) -> Self {
        Self {
            db,
            setting,
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

        let mut builder = make_smtp_builder(host, port as u16, encryption, None)?;
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

        // Check if the email service is actually configured.
        if smtp_server.read().await.is_none() {
            return Err(CoreError::EmailNotConfigured);
        }

        let from = self.from.read().await.clone();
        let address_from = to_address(from)?;
        let address_to = to_address(to)?;
        let msg = MessageBuilder::new()
            .subject(subject)
            .header(ContentType::TEXT_HTML)
            .from(Mailbox::new(Some("Tabby Admin".to_owned()), address_from))
            .to(Mailbox::new(None, address_to))
            .body(message)
            .map_err(anyhow::Error::msg)?;

        Ok(tokio::spawn(async move {
            if let Some(smtp_server) = &*(smtp_server.read().await) {
                match smtp_server.send(msg).await.map_err(anyhow::Error::msg) {
                    Ok(_) => {}
                    Err(err) => {
                        warn!("Failed to send mail due to {}", err);
                    }
                };
            }
        }))
    }
}

pub async fn new_email_service(db: DbConn) -> Result<impl EmailService> {
    let setting = db.read_email_setting().await?;
    let service = EmailServiceImpl::new(db.clone(), Arc::new(setting::create(db)));

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
    async fn read_setting(&self) -> Result<Option<EmailSetting>> {
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

    async fn update_setting(&self, input: EmailSettingInput) -> Result<()> {
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

    async fn delete_setting(&self) -> Result<()> {
        self.db.delete_email_setting().await?;
        // When the SMTP credentials are deleted, close the SMTP server connection
        self.shutdown_smtp_connection().await;
        Ok(())
    }

    async fn send_invitation(&self, email: String, code: String) -> Result<JoinHandle<()>> {
        let network_setting = self.setting.read_network_setting().await?;
        let external_url = network_setting.external_url;
        let body = templates::invitation(&external_url, &code, &email);
        self.send_email_in_background(
            email,
            "You've been invited to join a Tabby server!".into(),
            body,
        )
        .await
    }

    async fn send_signup(&self, email: String) -> Result<JoinHandle<()>> {
        let external_url = self.setting.read_network_setting().await?.external_url;

        let body = templates::signup_success(&external_url, &email);
        self.send_email_in_background(email, "Welcome to Tabby!".into(), body)
            .await
    }

    async fn send_password_reset(&self, email: String, code: String) -> Result<JoinHandle<()>> {
        let external_url = self.setting.read_network_setting().await?.external_url;
        let body = templates::password_reset(&external_url, &email, &code);
        self.send_email_in_background(email, "Reset your Tabby account password".into(), body)
            .await
    }

    async fn send_test(&self, to: String) -> Result<JoinHandle<()>> {
        let body = templates::test();
        self.send_email_in_background(to, "Your mail server is ready to go!".into(), body)
            .await
    }
}

fn to_address(email: String) -> anyhow::Result<Address> {
    let (user, domain) = email
        .split_once('@')
        .ok_or_else(|| anyhow!("address contains no @"))?;
    Ok(Address::new(user, domain)?)
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::{testutils::TestEmailServer, *};

    #[tokio::test]
    async fn test_update_email_with_service() {
        let db: DbConn = DbConn::new_in_memory().await.unwrap();
        let service = EmailServiceImpl::new(db.clone(), Arc::new(setting::create(db)));

        let update_input = EmailSettingInput {
            smtp_username: "test@example.com".into(),
            from_address: "test".into(),
            smtp_server: "smtp://example.com".into(),
            smtp_port: 578,
            encryption: Encryption::SslTls,
            auth_method: AuthMethod::None,
            smtp_password: Some("123456".to_owned()),
        };
        service.update_setting(update_input).await.unwrap();
        let setting = service.read_setting().await.unwrap().unwrap();
        assert_eq!(setting.smtp_username, "test@example.com");

        service.delete_setting().await.unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_send_email() {
        let mail_server = TestEmailServer::start().await;
        let service = mail_server
            .create_test_email_service(DbConn::new_in_memory().await.unwrap())
            .await;

        let handle = service
            .send_invitation("user@localhost".into(), "12345".into())
            .await
            .unwrap();

        handle.await.unwrap();

        let handle = service.send_test("user@localhost".into()).await.unwrap();

        handle.await.unwrap();

        let mails = mail_server.list_mail().await;
        let default_from = service.read_setting().await.unwrap().unwrap().from_address;
        assert!(mails[0].from.address.contains(&default_from));
    }

    #[tokio::test]
    #[serial]
    async fn test_send_test_email() {
        let mail_server = TestEmailServer::start().await;
        let service = mail_server
            .create_test_email_service(DbConn::new_in_memory().await.unwrap())
            .await;

        let handle = service.send_test("user@localhost".into()).await.unwrap();

        handle.await.unwrap();

        let mails = mail_server.list_mail().await;
        assert!(mails[0].subject.contains("ready to go"));
    }
}
