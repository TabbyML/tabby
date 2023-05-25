use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{response::Redirect, routing, Router, Server};
use hyper::Error;
use tower_http::cors::CorsLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod completions;
mod events;

#[derive(OpenApi)]
#[openapi(
    paths(events::log_event, completions::completion,),
    components(schemas(
        events::LogEventRequest,
        completions::CompletionRequest,
        completions::CompletionResponse,
        completions::Choice
    ))
)]
struct ApiDoc;

pub async fn main(model: &str) -> Result<(), Error> {
    let completions_state = Arc::new(completions::CompletionState::new(model));

    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/v1/events", routing::post(events::log_event))
        .route("/v1/completions", routing::post(completions::completion))
        .with_state(completions_state)
        .route(
            "/",
            routing::get(|| async { Redirect::temporary("/swagger-ui") }),
        )
        .layer(CorsLayer::permissive());

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080));
    println!("Listening at {}", address);
    Server::bind(&address).serve(app.into_make_service()).await
}
