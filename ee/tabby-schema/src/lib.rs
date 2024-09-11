//! Defines behavior for the tabby webserver which allows users to interact with enterprise features.
mod dao;
mod env;
mod schema;

pub mod juniper;
pub use dao::*;
pub use env::is_demo_mode;
pub use schema::*;
pub mod policy;

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
