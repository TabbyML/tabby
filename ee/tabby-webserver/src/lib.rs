//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
//! Using the web interface (e.g chat playground) requires using this module with the `--webserver` flag on the command line.

use std::sync::OnceLock;

use hash_ids::HashIds;
use juniper::ID;
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

static HASHER: OnceLock<HashIds> = OnceLock::new();

#[derive(thiserror::Error, Debug)]
#[error("Invalid ID")]
pub struct InvalidIDError;

fn hasher() -> &'static HashIds {
    HASHER.get_or_init(|| HashIds::builder().with_salt("tabby-id-serializer").finish())
}

pub fn to_id(rowid: i32) -> ID {
    ID::new(hasher().encode(&[rowid as u64]))
}

pub fn to_rowid(id: ID) -> Result<i32, InvalidIDError> {
    hasher()
        .decode(&id)
        .first()
        .map(|i| *i as i32)
        .ok_or(InvalidIDError)
}
