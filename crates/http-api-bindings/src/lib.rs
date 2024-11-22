mod chat;
mod completion;
mod embedding;

pub use chat::create as create_chat;
pub use completion::{build_completion_prompt, create};
pub use embedding::create as create_embedding;
use reqwest::Response;
use serde_json::Value;

#[async_trait::async_trait]
pub trait HttpClient: Send + Sync {
    async fn post(&self, url: &str, body: Value) -> anyhow::Result<Response>;
}

fn create_reqwest_client(api_endpoint: &str) -> reqwest::Client {
    let builder = reqwest::Client::builder();

    let is_localhost = api_endpoint.starts_with("http://localhost")
        || api_endpoint.starts_with("http://127.0.0.1");
    let builder = if is_localhost {
        builder.no_proxy()
    } else {
        builder
    };

    builder.build().unwrap()
}
