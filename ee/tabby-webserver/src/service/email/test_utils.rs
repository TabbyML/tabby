use std::time::Duration;

use serde::Deserialize;
use tabby_db::DbConn;
use tokio::process::{Child, Command};

use super::new_email_service;
use crate::schema::email::{AuthMethod, EmailService, EmailSettingInput, Encryption};

#[derive(Deserialize, Debug)]
pub struct Mail {
    pub sender: String,
    pub subject: String,
}

pub struct TestEmailServer {
    #[allow(unused)]
    child: Child,
}

impl TestEmailServer {
    pub async fn get_mail(&self) -> Vec<Mail> {
        let mails = reqwest::get("http://localhost:1080/api/messages")
            .await
            .unwrap();

        mails.json().await.unwrap()
    }
}

pub fn default_email_settings() -> EmailSettingInput {
    EmailSettingInput {
        smtp_username: "tabby".into(),
        smtp_server: "127.0.0.1".into(),
        smtp_port: 1025,
        from_address: "tabby@localhost".into(),
        encryption: Encryption::None,
        auth_method: AuthMethod::None,
        smtp_password: Some("fake".into()),
    }
}

pub async fn start_smtp_server() -> TestEmailServer {
    let mut cmd = Command::new("mailtutan");
    cmd.kill_on_drop(true);

    let child = cmd
        .spawn()
        .expect("You need to run `cargo install mailtutan` before running this test");
    tokio::time::sleep(Duration::from_secs(1)).await;
    TestEmailServer { child }
}

pub async fn setup_test_email_service() -> (impl EmailService, TestEmailServer) {
    let child = start_smtp_server().await;

    let db = DbConn::new_in_memory().await.unwrap();
    let service = new_email_service(db).await.unwrap();
    service
        .update_email_setting(default_email_settings())
        .await
        .unwrap();
    (service, child)
}
