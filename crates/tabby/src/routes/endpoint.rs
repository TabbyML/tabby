use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, Method, Request, Uri},
    middleware::Next,
    response::{IntoResponse, Response},
    Extension, Json,
};
use hyper::StatusCode;
use serde::Serialize;
use tabby_common::{axum::MaybeUserExt, config::EndpointConfig};
use tracing::error;

use super::rate_limit::EndpointRateLimiters;

pub async fn endpoint_policy(
    State(_config): State<Arc<EndpointConfig>>,
    Extension(MaybeUserExt(_user)): Extension<MaybeUserExt>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    next.run(request).await
}

#[derive(Serialize)]
pub struct EndpointInfo {
    pub name: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

pub async fn list_endpoints(State(config): State<Arc<EndpointConfig>>) -> Json<Vec<EndpointInfo>> {
    let endpoints = config
        .endpoints
        .iter()
        .map(|e| EndpointInfo {
            name: e.name.clone(),
            metadata: e.metadata.clone(),
        })
        .collect();
    Json(endpoints)
}

pub async fn endpoint(
    State((config, rate_limiters)): State<(Arc<EndpointConfig>, Arc<EndpointRateLimiters>)>,
    Extension(MaybeUserExt(user)): Extension<MaybeUserExt>,
    Path((name, path)): Path<(String, String)>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Response {
    let endpoint = config.endpoints.iter().find(|e| e.name == name);

    let endpoint = if let Some(endpoint) = endpoint {
        endpoint
    } else {
        return StatusCode::NOT_FOUND.into_response();
    };

    // User must be authenticated to access the endpoint
    let user = match user {
        Some(u) => u,
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };

    // Apply rate limiting if user_quota is configured
    if let Some(user_quota) = &endpoint.user_quota {
        let limiter = rate_limiters.get_or_create(
            &endpoint.name,
            &user.id.to_string(),
            user_quota.requests_per_minute,
        );
        // Try to acquire a permit, return 429 if rate limited
        if !limiter.try_acquire(1) {
            return StatusCode::TOO_MANY_REQUESTS.into_response();
        }
    }

    let client = reqwest::Client::new();
    let path = if path.starts_with('/') {
        path
    } else {
        format!("/{path}")
    };
    let query = uri.query().map(|q| format!("?{q}")).unwrap_or_default();
    let target_url = format!(
        "{}{}{}",
        endpoint.api_route.trim_end_matches('/'),
        path,
        query
    );
    let mut req = client.request(method, &target_url);

    if let Some(timeout) = endpoint.timeout {
        req = req.timeout(std::time::Duration::from_millis(timeout));
    }

    for (key, value) in headers {
        if let Some(key) = key {
            if key == axum::http::header::HOST {
                continue;
            }
            // Header allow duplicates, but it would cause errors, especially for authentication headers.
            // So, we are NOT copying headers that are already specified in `endpoint.headers`.
            if endpoint
                .headers
                .keys()
                .any(|k| k.eq_ignore_ascii_case(key.as_str()))
            {
                continue;
            }
            req = req.header(key, value);
        }
    }
    for (key, value) in &endpoint.headers {
        req = req.header(key, value);
    }

    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    req = req.body(bytes);

    match req.send().await {
        Ok(resp) => {
            // Forward the response back (including 4xx errors from upstream)
            let mut builder = Response::builder().status(resp.status());
            if let Some(headers) = builder.headers_mut() {
                *headers = resp.headers().clone();
            }
            builder
                .body(axum::body::Body::from_stream(resp.bytes_stream()))
                .unwrap_or_else(|e| {
                    error!(
                        "Failed to build response body for endpoint '{}': {}",
                        name, e
                    );
                    StatusCode::INTERNAL_SERVER_ERROR.into_response()
                })
        }
        Err(e) => {
            // Check if the error has an associated status code (e.g., from error_for_status())
            if let Some(status) = e.status() {
                error!(
                    "Endpoint '{}' ({}) returned error status {}: {}",
                    name, target_url, status, e
                );
                return status.into_response();
            }

            // Handle timeout errors with 504 Gateway Timeout
            if e.is_timeout() {
                error!(
                    "Request to endpoint '{}' ({}) timed out: {}",
                    name, target_url, e
                );
                return StatusCode::GATEWAY_TIMEOUT.into_response();
            }

            // Other errors (connection errors, etc.) return 500
            error!(
                "Failed to forward request to endpoint '{}' ({}): {}",
                name, target_url, e
            );
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}
