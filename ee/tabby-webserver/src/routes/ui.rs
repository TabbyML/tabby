use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
};

#[derive(rust_embed::RustEmbed)]
#[folder = "./ui"]
struct WebAssets;

struct WebStaticFile<T>(pub T);

impl<T> IntoResponse for WebStaticFile<T>
where
    T: Into<String>,
{
    fn into_response(self) -> Response {
        let make_404_response = || {
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from(WebAssets::get("404.html").unwrap().data))
                .unwrap_or_else(|_| panic!("Invalid response"))
        };

        let path = self.0.into();
        let Ok(decoded_path) = urlencoding::decode(&path) else {
            return make_404_response();
        };

        match WebAssets::get(decoded_path.as_ref()) {
            Some(content) => {
                let body = Body::from(content.data);
                let mime = mime_guess::from_path(&path).first_or_octet_stream();
                let mut builder =
                    Response::builder().header(header::CACHE_CONTROL, "public, max-age=604800");
                if !path.ends_with(".html") {
                    builder = builder.header(header::CONTENT_TYPE, mime.as_ref())
                };
                builder
                    .body(body)
                    .unwrap_or_else(|_| panic!("Invalid response"))
            }
            None => make_404_response(),
        }
    }
}

pub async fn handler(uri: Uri) -> impl IntoResponse {
    let mut path = uri.path().trim_start_matches('/').to_string();
    if path.is_empty() {
        path = "index.html".to_owned()
    } else if !path.contains('.') && WebAssets::get(&format!("{}.html", path)).is_some() {
        path += ".html"
    }
    WebStaticFile(path)
}
