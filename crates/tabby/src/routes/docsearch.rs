use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    Json,
};
use hyper::StatusCode;
use serde::Deserialize;
use tabby_common::api::doc::{DocSearch, DocSearchError, DocSearchResponse};
use tracing::{instrument, warn};
use utoipa::IntoParams;

#[derive(Deserialize, IntoParams)]
pub struct DocSearchQuery {
    #[param(default = "get")]
    q: String,

    #[param(default = 20)]
    limit: Option<usize>,

    #[param(default = 0)]
    offset: Option<usize>,
}

#[utoipa::path(
    get,
    params(DocSearchQuery),
    path = "/v1beta/docsearch",
    operation_id = "docsearch",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success" , body = DocSearchResponse, content_type = "application/json"),
        (status = 501, description = "When code search is not enabled, the endpoint will returns 501 Not Implemented"),
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state, query))]
pub async fn docsearch(
    State(state): State<Arc<dyn DocSearch>>,
    query: Query<DocSearchQuery>,
) -> Result<Json<DocSearchResponse>, StatusCode> {
    match state
        .search(
            &query.q,
            query.limit.unwrap_or(20),
            query.offset.unwrap_or(0),
        )
        .await
    {
        Ok(serp) => Ok(Json(serp)),
        Err(DocSearchError::NotReady) => Err(StatusCode::NOT_IMPLEMENTED),
        Err(DocSearchError::TantivyError(err)) => {
            warn!("{}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(DocSearchError::QueryParserError(err)) => {
            warn!("{}", err);
            Err(StatusCode::BAD_REQUEST)
        }
        Err(DocSearchError::Other(err)) => {
            warn!("{}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
