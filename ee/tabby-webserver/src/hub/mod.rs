pub mod api;
mod websocket;

use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use api::{ConnectHubRequest, Hub};
use axum::{
    extract::{ws::WebSocket, ConnectInfo, State, WebSocketUpgrade},
    response::IntoResponse,
    TypedHeader,
};
use hyper::{Body, StatusCode};
use juniper_axum::extract::AuthBearer;
use tabby_common::{
    api::{code::SearchResponse, event::LogEntry},
    config::RepositoryConfig,
};
use tarpc::server::{BaseChannel, Channel};
use tracing::warn;
use websocket::WebSocketTransport;

use crate::schema::ServiceLocator;

pub(crate) async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<dyn ServiceLocator>>,
    AuthBearer(token): AuthBearer,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    TypedHeader(request): TypedHeader<ConnectHubRequest>,
) -> impl IntoResponse {
    let unauthorized = axum::response::Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .body(Body::empty())
        .unwrap()
        .into_response();

    let Some(token) = token else {
        return unauthorized;
    };

    let Ok(registeration_token) = state.worker().read_registration_token().await else {
        return unauthorized;
    };

    if token != registeration_token {
        return unauthorized;
    }

    ws.on_upgrade(move |socket| handle_socket(state, socket, addr.ip(), request))
        .into_response()
}

async fn handle_socket(
    state: Arc<dyn ServiceLocator>,
    socket: WebSocket,
    addr: IpAddr,
    req: ConnectHubRequest,
) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let addr = match req {
        ConnectHubRequest::Scheduler => None,
        ConnectHubRequest::Worker(worker) => {
            let worker = worker.create_worker(addr);
            let addr = worker.addr.clone();
            match state.worker().register_worker(worker).await {
                Ok(_) => Some(addr),
                Err(err) => {
                    warn!("Failed to register worker: {}", err);
                    return;
                }
            }
        }
    };
    let imp = Arc::new(HubImpl::new(state.clone(), addr));
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}

struct HubImpl {
    ctx: Arc<dyn ServiceLocator>,
    worker_addr: Option<String>,
}

impl HubImpl {
    fn new(ctx: Arc<dyn ServiceLocator>, worker_addr: Option<String>) -> Self {
        Self { ctx, worker_addr }
    }
}

impl Drop for HubImpl {
    fn drop(&mut self) {
        let ctx = self.ctx.clone();
        if let Some(worker_addr) = self.worker_addr.clone() {
            tokio::spawn(async move {
                ctx.worker().unregister_worker(worker_addr.as_str()).await;
            });
        }
    }
}

#[tarpc::server]
impl Hub for Arc<HubImpl> {
    async fn write_log(self, _context: tarpc::context::Context, x: LogEntry) {
        self.ctx.logger().write(x)
    }

    async fn search(
        self,
        _context: tarpc::context::Context,
        q: String,
        limit: usize,
        offset: usize,
    ) -> SearchResponse {
        match self.ctx.code().search(&q, limit, offset).await {
            Ok(serp) => serp,
            Err(err) => {
                warn!("Failed to search: {}", err);
                SearchResponse::default()
            }
        }
    }

    async fn search_in_language(
        self,
        _context: tarpc::context::Context,
        git_url: String,
        language: String,
        tokens: Vec<String>,
        limit: usize,
        offset: usize,
    ) -> SearchResponse {
        match self
            .ctx
            .code()
            .search_in_language(&git_url, &language, &tokens, limit, offset)
            .await
        {
            Ok(serp) => serp,
            Err(err) => {
                warn!("Failed to search: {}", err);
                SearchResponse::default()
            }
        }
    }

    async fn list_repositories(self, _context: tarpc::context::Context) -> Vec<RepositoryConfig> {
        let mut repositories = vec![];

        let result = self
            .ctx
            .repository()
            .list_repositories(None, None, None, None)
            .await
            .map_err(|e| e.to_string())
            .map(|v| {
                v.into_iter()
                    .map(|r| RepositoryConfig::new_named(r.name, r.git_url))
                    .collect()
            });
        repositories.extend(result.unwrap_or_else(|e| {
            warn!("Failed to fetch repositories: {e}");
            vec![]
        }));

        let provider_service = self.ctx.github_repository_provider();
        let repository_providers = provider_service
            .list_github_repository_providers(None, None, None, None)
            .await
            .unwrap_or_else(|e| {
                warn!("Failed to fetch GitHub repository providers: {e}");
                vec![]
            });

        for provider in repository_providers {
            let Ok(access_token) = provider_service
                .read_github_repository_provider_access_token(provider.id.clone())
                .await
            else {
                continue;
            };

            let repos = match provider_service
                .list_github_provided_repositories_by_provider(
                    vec![provider.id.clone()],
                    None,
                    None,
                    None,
                    None,
                )
                .await
            {
                Ok(repos) => repos,
                Err(e) => {
                    warn!(
                        "Failed to retrieve repositories provided by {name}: {e}",
                        name = provider.display_name
                    );
                    continue;
                }
            };
            repositories.extend(repos.into_iter().filter(|repo| repo.active).map(|repo| {
                RepositoryConfig::new_named(
                    repo.name,
                    format_authenticated_git_url(repo.git_url, &access_token),
                )
            }))
        }

        repositories
    }
}

fn format_authenticated_git_url(mut git_url: String, access_token: &str) -> String {
    let split_pos = git_url.find("://").map(|i| i + 3).unwrap_or(0);
    git_url.insert_str(split_pos, &format!("{access_token}@"));
    git_url
}

#[cfg(test)]
mod tests {
    use crate::hub::format_authenticated_git_url;

    #[test]
    fn test_format_authenticated_git_url() {
        assert_eq!(
            format_authenticated_git_url("https://github.com/TabbyML/tabby".into(), "token"),
            "https://token@github.com/TabbyML/tabby"
        );
        assert_eq!(
            format_authenticated_git_url("github.com/TabbyML/tabby".into(), "token"),
            "token@github.com/TabbyML/tabby"
        );
    }
}
