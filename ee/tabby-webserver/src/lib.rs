mod handler;
mod hub;
mod oauth;
mod repositories;
mod schema;
mod service;
mod ui;

pub mod public {
    pub use super::{
        handler::attach_webserver,
        /* used by tabby workers (consumer of /hub api) */
        hub::api::{create_client, HubClient, RegisterWorkerRequest, WorkerKind},
        /* used by examples/update-schema.rs */ schema::create_schema,
    };
}
