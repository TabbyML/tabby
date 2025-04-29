//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
mod axum;
mod hub;
mod jwt;
mod ldap;
mod oauth;
mod path;
mod rate_limit;
mod routes;
mod service;
mod webserver;

#[cfg(test)]
pub use service::*;
use tabby_common::api;
use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    paths(
        routes::ingestion::ingestion,
        routes::ingestion::delete_ingestion_source,
        routes::ingestion::delete_ingestion,
    ),
    components(schemas(
        api::ingestion::IngestionRequest,
        api::ingestion::IngestionResponse,
    )),
    // modifiers(&SecurityAddon),
)]
pub struct EEApiDoc;

pub mod public {

    pub use super::{
        hub::{create_worker_client, WorkerClient},
        webserver::Webserver,
    };
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return std::result::Result::Err(anyhow::anyhow!($msg).into())
    };
    ($err:expr $(,)?) => {
        return std::result::Result::Err(anyhow::anyhow!($err).into())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return std::result::Result::Err(anyhow::anyhow!($fmt, $($arg)*).into())
    };
}
