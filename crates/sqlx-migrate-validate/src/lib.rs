//! With [sqlx] and its [sqlx::migrate!] macro it is possible to create migrations and apply them to a database.
//! When starting an application it is important to validate that the database is in the correct state.
//! This crate provides a way to validate that the applied migrations match the desired migrations.
//! In combination with the [sqlx::migrate!] macro it is possible to validate that the database schema
//! matches the migrations present in the source code at the time of compilation.
//!
//! While this does not ensure full compatibility between the database and the application it can help
//! to detect issues early.
//!
//! Examples:
//!
//! ```rust,no_run
//! use sqlx_migrate_validate::{Validate, Validator};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anyhow::Error> {
//!     let pool = sqlx::SqlitePool::connect("sqlite::memory:").await?;
//!     let mut conn = pool.acquire().await?;
//!
//!     sqlx::migrate!("./tests/migrations-1")
//!         .validate(&mut *conn)
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

mod error;
mod validate;

pub use error::ValidateError;
pub use validate::*;

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;
