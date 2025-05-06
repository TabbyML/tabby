use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use hyper::StatusCode;
use tabby_common::api::ingestion::{IngestionRequest, IngestionResponse};
use tabby_schema::ingestion::IngestionService;
use tracing::{error, instrument, warn};
use validator::Validate;

pub struct IngestionState {
    pub ingestion: Arc<dyn IngestionService>,
}

#[utoipa::path(
    post,
    path = "/v1beta/ingestion",
    operation_id = "ingestion",
    tag = "v1beta",
    responses(
        (status = 202, description = "Accepted, Waiting to be processed", content_type = "application/json"),
        (status = 400, description = "Bad Request"),
        (status = 401, description = "Unauthorized"),
        (status = 403, description = "Forbidden"),
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state, request))]
pub async fn ingestion(
    State(state): State<Arc<IngestionState>>,
    Json(request): Json<IngestionRequest>,
) -> Result<(StatusCode, Json<IngestionResponse>), StatusCode> {
    if let Err(e) = request.validate() {
        warn!("Invalid request: {}", e);
        return Err(StatusCode::BAD_REQUEST);
    }

    let response = state
        .ingestion
        .ingestion(request.clone())
        .await
        .map_err(|e| {
            error!("Failed to process ingestion: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok((StatusCode::ACCEPTED, Json(response)))
}

#[utoipa::path(
    delete,
    path = "/v1beta/ingestion/{source}",
    operation_id = "delete_ingestion_source",
    tag = "v1beta",
    responses(
        (status = 202, description = "Accepted, Waiting to be processed", content_type = "application/json"),
        (status = 400, description = "Bad Request"),
        (status = 401, description = "Unauthorized"),
        (status = 403, description = "Forbidden"),
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state))]
pub async fn delete_ingestion_source(
    State(state): State<Arc<IngestionState>>,
    Path(source): Path<String>,
) -> Result<StatusCode, StatusCode> {
    if source.is_empty() {
        warn!("Invalid request: source is empty");
        return Err(StatusCode::BAD_REQUEST);
    }

    state
        .ingestion
        .delete_by_source_id(source)
        .await
        .map_err(|e| {
            error!("Failed to process ingestion: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(StatusCode::ACCEPTED)
}

#[utoipa::path(
    delete,
    path = "/v1beta/ingestion/{source}/{id}",
    operation_id = "delete_ingestion",
    tag = "v1beta",
    responses(
        (status = 202, description = "Accepted, Waiting to be processed", content_type = "application/json"),
        (status = 400, description = "Bad Request"),
        (status = 401, description = "Unauthorized"),
        (status = 403, description = "Forbidden"),
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state))]
pub async fn delete_ingestion(
    State(state): State<Arc<IngestionState>>,
    Path((source, id)): Path<(String, String)>,
) -> Result<StatusCode, StatusCode> {
    if source.is_empty() || id.is_empty() {
        warn!("Invalid request: source or id is empty");
        return Err(StatusCode::BAD_REQUEST);
    }

    state.ingestion.delete(source, id).await.map_err(|e| {
        error!("Failed to process ingestion: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(StatusCode::ACCEPTED)
}
