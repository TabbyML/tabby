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
use schema::Schema;
use tracing::warn;

#[derive(Default)]
pub struct Webserver {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
}

impl Webserver {
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
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>, Arc<Webserver>>).with_state(ws.clone()),
        )
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
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
