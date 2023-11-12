//! [`JuniperResponse`] definition.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use juniper::{http::GraphQLBatchResponse, DefaultScalarValue, ScalarValue};

/// Wrapper around a [`GraphQLBatchResponse`], implementing [`IntoResponse`], so it can be returned
/// from [`axum`] handlers.
pub struct JuniperResponse<'a, S = DefaultScalarValue>(pub GraphQLBatchResponse<'a, S>)
where
    S: ScalarValue;

impl<S: ScalarValue> IntoResponse for JuniperResponse<'_, S> {
    fn into_response(self) -> Response {
        if self.0.is_ok() {
            Json(self.0).into_response()
        } else {
            (StatusCode::BAD_REQUEST, Json(self.0)).into_response()
        }
    }
}
