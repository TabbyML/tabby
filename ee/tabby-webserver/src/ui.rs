use axum::{
    body::{boxed, Full},
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
        let path = self.0.into();
        match WebAssets::get(path.as_str()) {
            Some(content) => {
                let body = boxed(Full::from(content.data));
                let mime = mime_guess::from_path(path).first_or_octet_stream();
                Response::builder()
                    .header(header::CONTENT_TYPE, mime.as_ref())
                    .body(body)
                    .unwrap_or_else(|_| panic!("Invalid response"))
            }
            None => Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(boxed(Full::from(WebAssets::get("404.html").unwrap().data)))
                .unwrap_or_else(|_| panic!("Invalid response")),
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
