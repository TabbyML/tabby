use async_trait::async_trait;
use axum::{body::Body, extract::Request, middleware::Next};

use crate::schema::Result;

#[async_trait]
pub trait WorkerService: Send + Sync {
    async fn read_registration_token(&self) -> Result<String>;
    async fn reset_registration_token(&self) -> Result<String>;

    async fn dispatch_request(
        &self,
        request: Request<Body>,
        next: Next,
    ) -> axum::response::Response;
    async fn is_chat_enabled(&self) -> Result<bool>;
}
