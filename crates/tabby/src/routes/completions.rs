use std::sync::Arc;

use axum::{extract::State, Json};
use axum_extra::{headers, TypedHeader};
use hyper::StatusCode;
use tabby_common::axum::MaybeUser;
use tracing::{instrument, warn};

use crate::services::completion::{CompletionRequest, CompletionResponse, CompletionService};

#[utoipa::path(
    post,
    path = "/v1/completions",
    request_body = CompletionRequest,
    operation_id = "completion",
    tag = "v1",
    responses(
        (status = 200, description = "Success", body = CompletionResponse, content_type = "application/json"),
        (status = 400, description = "Bad Request")
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state, request))]
pub async fn completions(
    State(state): State<Arc<CompletionService>>,
    TypedHeader(MaybeUser(user)): TypedHeader<MaybeUser>,
    TypedHeader(user_agent): TypedHeader<headers::UserAgent>,
    Json(mut request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    if let Some(user) = user {
        request.user.replace(user);
    }

    match state.generate(&request, &user_agent.to_string()).await {
        Ok(resp) => Ok(Json(resp)),
        Err(err) => {
            warn!("{}", err);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}
