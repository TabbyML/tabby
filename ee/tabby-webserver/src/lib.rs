//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
mod axum;
mod hub;
mod jwt;
mod oauth;
mod path;
mod routes;
mod service;
mod webserver;

#[cfg(test)]
pub use service::*;

pub mod public {

    pub use super::{
        /* used by tabby workers (consumer of /hub api) */
        hub::{
            create_scheduler_client, create_worker_client, RegisterWorkerRequest, SchedulerClient,
            WorkerClient, WorkerKind,
        },
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
