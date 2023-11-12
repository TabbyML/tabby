mod proxy;
mod schema;
mod ui;
mod worker;

use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::State,
    http::Request,
    middleware::{from_fn_with_state, Next},
    response::IntoResponse,
    routing, Extension, Router,
};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use juniper_axum::{graphiql, graphql, playground};
use schema::{Schema, WorkerKind};
use thiserror::Error;
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum WebserverError {
    #[error("Invalid worker token")]
    InvalidToken(String),

    #[error("Feature requires enterprise license")]
    RequiresEnterpriseLicense,
}

#[derive(Default)]
pub struct Webserver {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
}

// FIXME: generate token and support refreshing in database.
static WORKER_TOKEN: &str = "4c749fad-2be7-45a3-849e-7714ccade382";

impl Webserver {
    async fn register_worker(
        &self,
        token: String,
        client_addr: SocketAddr,
        kind: WorkerKind,
        port: i32,
    ) -> Result<String, WebserverError> {
        if token != WORKER_TOKEN {
            return Err(WebserverError::InvalidToken(token));
        }

        let addr = SocketAddr::new(client_addr.ip(), port as u16);
        let addr = match kind {
            WorkerKind::Completion => self.completion.register(addr).await,
            WorkerKind::Chat => self.chat.register(addr).await,
        };

        if let Some(addr) = addr {
            info!("registering <{:?}> worker running at {}", kind, addr);
            Ok(addr)
        } else {
            Err(WebserverError::RequiresEnterpriseLicense)
        }
    }

    async fn dispatch_request(
        &self,
        request: Request<Body>,
        next: Next<Body>,
    ) -> axum::response::Response {
        let path = request.uri().path();

        let remote_addr = request
            .extensions()
            .get::<axum::extract::ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0)
            .expect("Unable to extract remote addr");

        let worker = if path.starts_with("/v1/completions") {
            self.completion.select().await
        } else if path.starts_with("/v1beta/chat/completions") {
            self.chat.select().await
        } else {
            None
        };

        if let Some(worker) = worker {
            match proxy::call(self.client.clone(), remote_addr.ip(), &worker, request).await {
                Ok(res) => res.into_response(),
                Err(err) => {
                    warn!("Failed to proxy request {}", err);
                    axum::response::Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .unwrap()
                        .into_response()
                }
            }
        } else {
            next.run(request).await
        }
    }
}

pub fn attach_webserver(router: Router) -> Router {
    let ws = Arc::new(Webserver::default());
    let schema = Arc::new(schema::new());

    let app = Router::new()
        .route("/graphql", routing::get(playground("/graphql", None)))
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>, Arc<Webserver>>).with_state(ws.clone()),
        )
        .layer(Extension(schema));

    router
        .merge(app)
        .fallback(ui::handler)
        .layer(from_fn_with_state(ws, distributed_tabby_layer))
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<Webserver>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.dispatch_request(request, next).await
}
