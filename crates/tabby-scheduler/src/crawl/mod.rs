mod types;

use std::process::Stdio;

use async_stream::stream;
use futures::{Stream, StreamExt};
use readable_readability::Readability;
use tokio::io::AsyncBufReadExt;
use tracing::{debug, warn};
use url::Url;

use self::types::{CrawledDocument, KatanaRequestResponse};

async fn crawl_url(start_url: &str) -> impl Stream<Item = KatanaRequestResponse> {
    let mut child = tokio::process::Command::new("katana")
        .arg("-u")
        .arg(start_url)
        .arg("-jsonl")
        .arg("-mdc")
        .arg(format!("starts_with(endpoint, \"{start_url}\")"))
        .arg("-depth")
        .arg("9999")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start katana, please check whether the binary is in your $PATH");

    let stdout = child.stdout.take().expect("Failed to acquire stdout");
    let mut stdout = tokio::io::BufReader::new(stdout).lines();

    tokio::spawn(async move {
        if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
            warn!("Katana exited with code {}", exit_code);
        }
    });

    stream! {
        while let Ok(Some(line)) = stdout.next_line().await {
            let Ok(data) = serde_json::from_str::<KatanaRequestResponse>(&line) else {
                warn!("Failed to parse katana output, skipping...");
                continue;
            };


            yield data;
        }
    }
}

async fn to_document(data: KatanaRequestResponse) -> Option<CrawledDocument> {
    // Skip if the status code is not 200
    if data.response.status_code != 200 {
        return None;
    }

    // Skip if the content type is not text/html
    if !data
        .response
        .headers
        .get("content_type")
        .is_some_and(|ct| ct.starts_with("text/html"))
    {
        return None;
    }

    // Cleanup the HTML with Readability
    let (html, metadata) = {
        let (node, metadata) = Readability::new()
            .base_url(Url::parse(&data.request.endpoint).ok()?)
            .parse(&data.response.body);

        let mut html_bytes = vec![];
        node.serialize(&mut html_bytes).ok()?;
        (String::from_utf8(html_bytes).ok()?, metadata)
    };

    // Convert the HTML to Markdown
    let md = mdka::from_html(&html);

    // Remove any HTML tags that might have been left behind
    let md = voca_rs::strip::strip_tags(&md).trim().to_owned();

    // Skip if the document is empty
    if md.is_empty() {
        return None;
    }

    Some(CrawledDocument::new(
        data.request.endpoint,
        md,
        metadata.into(),
    ))
}

pub async fn crawl_pipeline(start_url: &str) -> impl Stream<Item = CrawledDocument> {
    crawl_url(start_url).await.filter_map(to_document)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tracing_test::traced_test;

    use super::*;

    #[tokio::test]
    #[traced_test]
    async fn test_to_document() {
        let headers = vec![("content_type".into(), "text/html".into())]
            .iter()
            .cloned()
            .collect();
        let data = KatanaRequestResponse {
            timestamp: "2021-09-01T00:00:00Z".to_owned(),
            request: types::KatanaRequest {
                endpoint: "https://example.com".to_owned(),
                method: "GET".to_owned(),
                raw: "GET / HTTP/1.1\nHost: example.com\n".to_owned(),
            },
            response: types::KatanaResponse {
                status_code: 200,
                headers,
                body: "<p>Hello, World!</p>".to_owned(),
                technologies: Default::default(),
                raw: "HTTP/1.1 200 OK\nContent-Type: text/html\n".to_owned(),
            },
        };

        let doc = to_document(data).await.unwrap();
        assert_eq!(doc.url, "https://example.com");
        assert_eq!(doc.markdown, "Hello, World!");
    }
}
