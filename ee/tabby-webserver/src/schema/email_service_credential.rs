use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;
use tabby_db::EmailServiceCredentialDAO;

#[derive(GraphQLObject)]
pub struct EmailServiceCredential {
    pub smtp_username: String,
    pub smtp_server: String,
}

#[async_trait]
pub trait EmailServiceCredentialService: Send + Sync {
    async fn get_email_service_credential(&self) -> Result<Option<EmailServiceCredential>>;
    async fn update_email_service_credential(&self, creds: EmailServiceCredentialDAO)
        -> Result<()>;
    async fn delete_email_service_credential(&self) -> Result<()>;
}
