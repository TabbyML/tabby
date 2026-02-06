use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, Method, Request, Uri},
    middleware::Next,
    response::{IntoResponse, Response},
    Extension,
};
use hyper::StatusCode;
use tabby_common::{axum::MaybeUserExt, config::AgentConfig};

pub async fn agent_policy(
    State(_config): State<Arc<AgentConfig>>,
    Extension(MaybeUserExt(_user)): Extension<MaybeUserExt>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    next.run(request).await
}

pub async fn endpoint(
    State(config): State<Arc<AgentConfig>>,
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

    let client = reqwest::Client::new();
    let path = if path.starts_with('/') {
        path
    } else {
        format!("/{}", path)
    };
    let query = uri.query().map(|q| format!("?{}", q)).unwrap_or_default();
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
            let mut builder = Response::builder().status(resp.status());
            if let Some(headers) = builder.headers_mut() {
                *headers = resp.headers().clone();
            }
            builder
                .body(axum::body::Body::from_stream(resp.bytes_stream()))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
