//! Types and traits for extracting data from [`Request`]s.

use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{request::Parts, HeaderValue, StatusCode},
};

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
