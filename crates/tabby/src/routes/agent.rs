use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, Method, Request, Uri},
    middleware::Next,
    response::{IntoResponse, Response},
    Extension,
};
use hyper::StatusCode;
use tabby_common::{axum::MaybeUserExt, config::AgentConfig};

pub async fn agent_policy(
    State(config): State<Arc<AgentConfig>>,
    Extension(MaybeUserExt(user)): Extension<MaybeUserExt>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let groups = user.as_ref().map(|u| u.groups.as_slice()).unwrap_or(&[]);
    let is_allowed = config.tools.is_empty()
        || config.tools.iter().any(|tool| {
            tool.available_groups
                .as_ref()
                .map(|allowed| allowed.iter().any(|group| groups.contains(group)))
                .unwrap_or(true)
        });

    if !is_allowed {
        return StatusCode::FORBIDDEN.into_response();
    }

    next.run(request).await
}

pub async fn agent(
    State(config): State<Arc<AgentConfig>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Response {
    let path = uri.path();
    let (scheme, stripped_path) = if let Some(p) = path.strip_prefix("/v2alpha/agent/https/") {
        ("https", p)
    } else if let Some(p) = path.strip_prefix("/v2alpha/agent/http/") {
        ("http", p)
    } else {
        return StatusCode::BAD_REQUEST.into_response();
    };

    let query = uri.query().map(|q| format!("?{}", q)).unwrap_or_default();
    let target_url = format!("{}://{}{}", scheme, stripped_path, query);

    if !target_url.starts_with(&config.api_endpoint) {
        return StatusCode::FORBIDDEN.into_response();
    }

    let client = reqwest::Client::new();
    let mut req = client.request(method, &target_url);

    for (key, value) in headers {
        if let Some(key) = key {
            if key == axum::http::header::HOST {
                continue;
            }
            req = req.header(key, value);
        }
    }

    for (key, value) in &config.headers {
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
