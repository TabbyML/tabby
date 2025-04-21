//! Types and traits for extracting data from [`Request`]s.

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, HeaderValue, StatusCode},
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AuthBearer(pub Option<String>);

pub type Rejection = (StatusCode, &'static str);

impl<B> FromRequestParts<B> for AuthBearer
where
    B: Send + Sync,
{
    type Rejection = Rejection;

    async fn from_request_parts(req: &mut Parts, _: &B) -> Result<Self, Self::Rejection> {
        let access_token = req.uri.query().and_then(|q| {
            querystring::querify(q)
                .into_iter()
                .filter(|(k, _)| *k == "access_token")
                .map(|(_, v)| v)
                .next()
        });

        if let Some(access_token) = access_token {
            return Ok(Self(Some(access_token.into())));
        };

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
        Ok(Self(extract_bearer_token(authorization)))
    }
}

pub fn extract_bearer_token(authorization: &str) -> Option<String> {
    let split = authorization.split_once(' ');
    match split {
        Some(("Bearer", contents)) => Some(contents.to_owned()),
        _ => None,
    }
}
