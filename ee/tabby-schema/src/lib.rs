//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
//! Using the web interface (e.g chat playground) requires using this module with the `--webserver` flag on the command line.
mod dao;
mod env;
mod schema;

pub mod juniper;
pub use dao::*;
pub use env::demo_mode;
pub use schema::*;

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
