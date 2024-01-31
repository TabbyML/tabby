use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;

#[derive(GraphQLObject)]
pub struct EmailSetting {
    pub smtp_username: String,
    pub smtp_server: String,
}

#[async_trait]
pub trait EmailService: Send + Sync {
    async fn get_email_setting(&self) -> Result<Option<EmailSetting>>;
    async fn update_email_setting(
        &self,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<()>;
    async fn delete_email_setting(&self) -> Result<()>;
    async fn send_mail(&self, to: String, subject: String, body: String) -> Result<()>;

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
