//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
//! Using the web interface (e.g chat playground) requires using this module with the `--webserver` flag on the command line.
mod axum;
mod env;
mod handler;
mod hub;
mod juniper;
mod oauth;
mod path;
mod repositories;
mod schema;
mod service;
mod ui;

#[cfg(test)]
pub use service::*;

pub mod public {

    pub use super::{
        handler::WebserverHandle,
        /* used by tabby workers (consumer of /hub api) */
        hub::api::{
            create_scheduler_client, create_worker_client, RegisterWorkerRequest, SchedulerClient,
            WorkerClient, WorkerKind,
        },
        /* used by examples/update-schema.rs */
        schema::create_schema,
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

#[macro_export]
macro_rules! warn_stderr {
    ($ctx:expr, $($params:tt)+) => {
        tracing::warn!($($params)+);
        $ctx.stderr_writeline(format!($($params)+)).await;
    }
}
