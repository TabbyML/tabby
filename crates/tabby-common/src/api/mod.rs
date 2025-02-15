pub mod code;
pub mod commit;
pub mod event;
pub mod server_setting;
pub mod structured_doc;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, SearchError>;

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("index not ready")]
    NotReady,

    #[error(transparent)]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error(transparent)]
    TantivyError(#[from] tantivy::TantivyError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
