//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
//! Using the web interface (e.g chat playground) requires using this module with the `--webserver` flag on the command line.

mod handler;
mod hub;
mod oauth;
mod repositories;
mod schema;
mod service;
mod ui;

pub mod public {

    pub static USER_HEADER_FIELD_NAME: HeaderName = HeaderName::from_static("x-tabby-user");

    use axum::http::HeaderName;

    pub use super::{
        handler::attach_webserver,
        /* used by tabby workers (consumer of /hub api) */
        hub::api::{
            create_client, ConnectHubRequest, HubClient, RegisterWorkerRequest, RepositoryAccess,
            WorkerKind,
        },
        /* used by examples/update-schema.rs */ schema::create_schema,
    };
}
