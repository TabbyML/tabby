use std::sync::Arc;

use async_openai::types::Choice;
use axum::{extract::State, http::request, Extension, Json};
use axum_extra::{headers, TypedHeader};
use hyper::StatusCode;
use tabby_common::axum::{AllowedCodeRepository, MaybeUser};
use tantivy::Segment;
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
    Extension(allowed_code_repository): Extension<AllowedCodeRepository>,
    TypedHeader(MaybeUser(user)): TypedHeader<MaybeUser>,
    user_agent: Option<TypedHeader<headers::UserAgent>>,
    Json(mut request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    if let Some(user) = user {
        request.user.replace(user);
    }

    let user_agent = user_agent.map(|x| x.0.to_string());

    let mut use_crlf = false;
    if let Some(segments) = request.segments {
        let mut new_segments = segments.clone();
        if segments.prefix.contains("\r\n") {
            use_crlf = true;
            new_segments.prefix = segments.prefix.replace("\r\n", "\n");
        }
        if let Some(suffix) = segments.suffix {
            if suffix.contains("\r\n") {
                use_crlf = true;
                new_segments.suffix = Some(suffix.replace("\r\n", "\n"));
            }
        }
        request.segments = Some(new_segments);
    }

    match state
        .generate(&request, &allowed_code_repository, user_agent.as_deref())
        .await
    {
        Ok(resp) => {
            if use_crlf {
                let mut response_crlf = resp.clone();
                for (index, choice) in resp.choices.iter().enumerate() {
                    response_crlf.choices[index].text = choice.text.replace("\n", "\r\n");
                }

                return Ok(Json(response_crlf));
            }

            Ok(Json(resp))
        }
        Err(err) => {
            warn!("{}", err);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}
