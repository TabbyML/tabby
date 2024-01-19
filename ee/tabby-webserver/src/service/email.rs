use anyhow::{anyhow, Result};
use async_trait::async_trait;
use lettre::{transport::smtp::authentication::Credentials, Message, SmtpTransport, Transport};
use tabby_db::DbConn;
use tokio::sync::RwLock;

use crate::schema::email::{EmailService, EmailSetting};

struct EmailServiceImpl {
    db: DbConn,
    smtp_server: RwLock<Option<SmtpTransport>>,
}

impl EmailServiceImpl {
    /// (Re)initialize the SMTP server connection using new credentials
    async fn reset_smtp_connection(
        &self,
        username: String,
        password: String,
        server: &str,
    ) -> Result<()> {
        let mut smtp_server = self.smtp_server.write().await;
        *smtp_server = Some(
            SmtpTransport::relay(server)?
                .credentials(Credentials::new(username, password))
                .build(),
        );
        Ok(())
    }

    /// Close the SMTP server connection
    async fn shutdown_smtp_connection(&self) {
        *self.smtp_server.write().await = None;
    }
}

pub async fn new_email_service(db: DbConn) -> Result<impl EmailService> {
    let creds = db.read_email_setting().await?;
    let service = EmailServiceImpl {
        db,
        smtp_server: Default::default(),
    };
    // Optionally initialize the SMTP connection when the service is created
    if let Some(creds) = creds {
        service
            .reset_smtp_connection(creds.smtp_username, creds.smtp_password, &creds.smtp_server)
            .await?;
    };
    Ok(service)
}

#[async_trait]
impl EmailService for EmailServiceImpl {
    async fn get_email_setting(&self) -> Result<Option<EmailSetting>> {
        let creds = self.db.read_email_setting().await?;
        Ok(creds.map(Into::into))
    }

    async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()> {
        self.update_email_setting(
            smtp_username.clone(),
            smtp_password.clone(),
            smtp_server.clone(),
        )
        .await?;
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
        self.reset_smtp_connection(smtp_username, smtp_password, &smtp_server)
            .await?;
        Ok(())
    }

    async fn delete_email_setting(&self) -> Result<()> {
        self.delete_email_setting().await?;
        // When the SMTP credentials are deleted, close the SMTP server connection
        self.shutdown_smtp_connection().await;
        Ok(())
    }

    async fn send_mail(&self, message: &Message) -> Result<()> {
        let smtp_server = self.smtp_server.read().await;
        let Some(smtp_server) = &*smtp_server else {
            return Err(anyhow!("email settings have not been populated"));
        };
        smtp_server.send(message)?;
        Ok(())
    }
}
