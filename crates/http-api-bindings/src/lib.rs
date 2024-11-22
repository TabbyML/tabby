mod chat;
mod completion;
mod embedding;

use std::{env, sync::Arc, time::Duration};

pub use chat::create as create_chat;
pub use completion::{build_completion_prompt, create};
pub use embedding::create as create_embedding;
use reqwest::{Response, Url};
use serde::Serialize;
use tokio::sync::Mutex;
use tower::{limit::rate::RateLimit, Service, ServiceBuilder, ServiceExt};

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

struct RateLimitedClient {
    client: reqwest::Client,
    rate_limit: Arc<Mutex<RateLimit<reqwest::Client>>>,

    api_key: Option<String>,

    rate_limit_enabled: bool,
}

impl RateLimitedClient {
    pub fn new(
        api_endpoint: &str,
        api_key: Option<String>,
        num_request: u64,
        per: Duration,
    ) -> anyhow::Result<Self> {
        if (num_request == 0) != (per.as_secs() == 0) {
            anyhow::bail!("Both num_request and per must be zero or both must be non-zero");
        }

        let rate_limit_enabled = num_request > 0;

        let builder = reqwest::Client::builder();
        let is_localhost = api_endpoint.starts_with("http://localhost")
            || api_endpoint.starts_with("http://127.0.0.1");
        let builder = if is_localhost {
            builder.no_proxy()
        } else {
            builder
        };

        const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
        let client = builder.user_agent(USER_AGENT).tcp_nodelay(false).build()?;

        Ok(Self {
            client: client.clone(),
            rate_limit: Arc::new(Mutex::new(
                ServiceBuilder::new()
                    .rate_limit(num_request, per)
                    .service(client),
            )),
            api_key,
            rate_limit_enabled,
        })
    }

    async fn post<T: Serialize>(&self, url: &str, body: T) -> anyhow::Result<Response> {
        let url = Url::parse(url)?;
        let mut builder = self.client.post(url.clone()).json(&body);
        if let Some(api_key) = &self.api_key {
            builder = builder.bearer_auth(api_key);
        }
        let request = builder.build()?;

        if self.rate_limit_enabled {
            let future = self.rate_limit.lock().await.ready().await?.call(request);
            future
                .await
                .and_then(|response| response.error_for_status())
                .map_err(|err| anyhow::anyhow!("Error from server: {}", err))
        } else {
            let response = self.client.execute(request).await?;
            response
                .error_for_status()
                .map_err(|err| anyhow::anyhow!("Error from server: {}", err))
        }
    }
}
