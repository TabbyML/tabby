use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use axum_extra::TypedHeader;
use futures::{Stream, StreamExt};
use tabby_common::axum::MaybeUser;
use tracing::instrument;

use crate::service::answer::{AnswerRequest, AnswerService};

#[utoipa::path(
    post,
    request_body = AnswerRequest,
    path = "/v1beta/answer",
    operation_id = "answer",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success" , body = AnswerResponseChunk, content_type = "text/event-stream"),
        (status = 501, description = "When answer search is not enabled, the endpoint will returns 501 Not Implemented"),
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state, request))]
pub async fn answer(
    State(state): State<Arc<AnswerService>>,
    TypedHeader(MaybeUser(user)): TypedHeader<MaybeUser>,
    Json(mut request): Json<AnswerRequest>,
) -> Sse<impl Stream<Item = Result<Event, serde_json::Error>>> {
    if let Some(user) = user {
        request.user.replace(user);
    }

    let stream = state.answer(request).await;
    Sse::new(stream.map(|chunk| match serde_json::to_string(&chunk) {
        Ok(s) => Ok(Event::default().data(s)),
        Err(err) => Err(err),
    }))
    .keep_alive(KeepAlive::default())
}
