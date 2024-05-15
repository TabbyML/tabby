use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    Json,
};
use hyper::StatusCode;
use serde::Deserialize;
use tabby_common::api::code::{CodeSearch, CodeSearchError, CodeSearchResponse};
use tracing::{instrument, warn};
use utoipa::IntoParams;

#[derive(Deserialize, IntoParams)]
pub struct CodeSearchQuery {
    #[param(default = "get")]
    q: String,

    #[param(default = 20)]
    limit: Option<usize>,

    #[param(default = 0)]
    offset: Option<usize>,
}

#[utoipa::path(
    get,
    params(CodeSearchQuery),
    path = "/v1beta/search",
    operation_id = "search",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success" , body = CodeSearchResponse, content_type = "application/json"),
        (status = 501, description = "When code search is not enabled, the endpoint will returns 501 Not Implemented"),
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state, query))]
pub async fn search(
    State(state): State<Arc<dyn CodeSearch>>,
    query: Query<CodeSearchQuery>,
) -> Result<Json<CodeSearchResponse>, StatusCode> {
    match state
        .search(
            &query.q,
            query.limit.unwrap_or(20),
            query.offset.unwrap_or(0),
        )
        .await
    {
        Ok(serp) => Ok(Json(serp)),
        Err(CodeSearchError::NotReady) => Err(StatusCode::NOT_IMPLEMENTED),
        Err(CodeSearchError::TantivyError(err)) => {
            warn!("{}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(CodeSearchError::QueryParserError(err)) => {
            warn!("{}", err);
            Err(StatusCode::BAD_REQUEST)
        }
        Err(CodeSearchError::Other(err)) => {
            warn!("{}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
