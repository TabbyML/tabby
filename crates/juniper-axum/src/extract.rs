//! Types and traits for extracting data from [`Request`]s.

use std::fmt;

use axum::{
    async_trait,
    body::Body,
    extract::{FromRequest, FromRequestParts, Query},
    http::{request::Parts, HeaderValue, Method, Request, StatusCode},
    response::{IntoResponse as _, Response},
    Json, RequestExt as _,
};
use juniper::{
    http::{GraphQLBatchRequest, GraphQLRequest},
    DefaultScalarValue, ScalarValue,
};
use serde::Deserialize;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AuthBearer(pub Option<String>);

pub type Rejection = (StatusCode, &'static str);

#[async_trait]
impl<B> FromRequestParts<B> for AuthBearer
where
    B: Send + Sync,
{
    type Rejection = Rejection;

    async fn from_request_parts(req: &mut Parts, _: &B) -> Result<Self, Self::Rejection> {
        // Get authorization header
        let authorization = req
            .headers
            .get("authorization")
            .map(HeaderValue::to_str)
            .transpose()
            .map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    "authorization contains invalid characters",
                )
            })?;

        let Some(authorization) = authorization else {
            return Ok(Self(None));
        };

        // Check that its a well-formed bearer and return
        let split = authorization.split_once(' ');
        match split {
            // Found proper bearer
            Some(("Bearer", contents)) => Ok(Self(Some(contents.to_owned()))),
            _ => Ok(Self(None)),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct JuniperRequest<S = DefaultScalarValue>(pub GraphQLBatchRequest<S>)
where
    S: ScalarValue;

#[async_trait]
impl<S, State> FromRequest<State, Body> for JuniperRequest<S>
where
    S: ScalarValue,
    State: Sync,
    Query<GetRequest>: FromRequestParts<State>,
    Json<GraphQLBatchRequest<S>>: FromRequest<State, Body>,
    <Json<GraphQLBatchRequest<S>> as FromRequest<State, Body>>::Rejection: fmt::Display,
    String: FromRequest<State, Body>,
{
    type Rejection = Response;

    async fn from_request(mut req: Request<Body>, state: &State) -> Result<Self, Self::Rejection> {
        let content_type = req
            .headers()
            .get("content-type")
            .map(HeaderValue::to_str)
            .transpose()
            .map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    "`Content-Type` header is not a valid HTTP header string",
                )
                    .into_response()
            })?;

        match (req.method(), content_type) {
            (&Method::GET, _) => req
                .extract_parts::<Query<GetRequest>>()
                .await
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Invalid request query string: {e}"),
                    )
                        .into_response()
                })
                .and_then(|query| {
                    query
                        .0
                        .try_into()
                        .map(|q| Self(GraphQLBatchRequest::Single(q)))
                        .map_err(|e| {
                            (
                                StatusCode::BAD_REQUEST,
                                format!("Invalid request query `variables`: {e}"),
                            )
                                .into_response()
                        })
                }),
            (&Method::POST, Some("application/json")) => {
                Json::<GraphQLBatchRequest<S>>::from_request(req, state)
                    .await
                    .map(|req| Self(req.0))
                    .map_err(|e| {
                        (StatusCode::BAD_REQUEST, format!("Invalid JSON body: {e}")).into_response()
                    })
            }
            (&Method::POST, Some("application/graphql")) => String::from_request(req, state)
                .await
                .map(|body| {
                    Self(GraphQLBatchRequest::Single(GraphQLRequest::new(
                        body, None, None,
                    )))
                })
                .map_err(|_| (StatusCode::BAD_REQUEST, "Not valid UTF-8 body").into_response()),
            (&Method::POST, _) => Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "`Content-Type` header is expected to be either `application/json` or \
                 `application/graphql`",
            )
                .into_response()),
            _ => Err((
                StatusCode::METHOD_NOT_ALLOWED,
                "HTTP method is expected to be either GET or POST",
            )
                .into_response()),
        }
    }
}

/// Workaround for a [`GraphQLRequest`] not being [`Deserialize`]d properly from a GET query string,
/// containing `variables` in JSON format.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct GetRequest {
    query: String,
    #[serde(rename = "operationName")]
    operation_name: Option<String>,
    variables: Option<String>,
}

impl<S: ScalarValue> TryFrom<GetRequest> for GraphQLRequest<S> {
    type Error = serde_json::Error;
    fn try_from(req: GetRequest) -> Result<Self, Self::Error> {
        let GetRequest {
            query,
            operation_name,
            variables,
        } = req;
        Ok(Self::new(
            query,
            operation_name,
            variables.map(|v| serde_json::from_str(&v)).transpose()?,
        ))
    }
}
