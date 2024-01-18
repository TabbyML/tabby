use std::sync::RwLock;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use lettre::{transport::smtp::authentication::Credentials, Message, SmtpTransport, Transport};
use tabby_db::DbConn;

use crate::schema::email::{EmailService, EmailSetting};

struct EmailServiceImpl {
    db: DbConn,
    smtp_server: RwLock<Option<SmtpTransport>>,
}

impl EmailServiceImpl {
    fn init_server(&self, username: String, password: String, server: &str) -> Result<()> {
        let mut smtp_server = self.smtp_server.write().unwrap();
        *smtp_server = Some(
            SmtpTransport::relay(server)?
                .credentials(Credentials::new(username, password))
                .build(),
        );
        Ok(())
    }

    fn close_smtp_connection(&self) {
        *self.smtp_server.write().unwrap() = None;
    }
}

pub async fn new_email_service(db: DbConn) -> Result<impl EmailService> {
    let creds = db.read_email_setting().await?;
    let service = EmailServiceImpl {
        db,
        smtp_server: Default::default(),
    };
    if let Some(creds) = creds {
        service.init_server(creds.smtp_username, creds.smtp_password, &creds.smtp_server)?;
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
        self.init_server(smtp_username, smtp_password, &smtp_server)?;
        Ok(())
    }

    async fn delete_email_setting(&self) -> Result<()> {
        self.delete_email_setting().await?;
        self.close_smtp_connection();
        Ok(())
    }

    async fn send_mail(&self, message: &Message) -> Result<()> {
        let smtp_server = self.smtp_server.read().unwrap();
        let Some(smtp_server) = &*smtp_server else {
            return Err(anyhow!("email settings have not been populated"));
        };
        smtp_server.send(message)?;
        Ok(())
    }
}
