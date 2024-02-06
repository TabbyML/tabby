use std::sync::Arc;

use axum::{extract::State, headers::Header, http::HeaderName, Json, TypedHeader};
use hyper::StatusCode;
use tabby_common::constants::USER_HEADER_FIELD_NAME;
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
    Json(mut request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    if let Some(user) = user {
        request.user.replace(user);
    }
    match state.generate(&request).await {
        Ok(resp) => Ok(Json(resp)),
        Err(err) => {
            warn!("{}", err);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}

#[derive(Debug)]
pub struct MaybeUser(Option<String>);

pub static USER_HEADER: HeaderName = HeaderName::from_static(USER_HEADER_FIELD_NAME);

impl Header for MaybeUser {
    fn name() -> &'static axum::http::HeaderName {
        &USER_HEADER
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, axum::headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i axum::http::HeaderValue>,
    {
        let Some(value) = values.next() else {
            return Ok(MaybeUser(None));
        };
        let str = value.to_str().expect("User email is always a valid string");
        Ok(MaybeUser(Some(str.to_string())))
    }

    fn encode<E: Extend<axum::http::HeaderValue>>(&self, _values: &mut E) {
        todo!()
    }
}
