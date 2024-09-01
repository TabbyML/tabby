mod types;

use std::process::Stdio;

use async_stream::stream;
use futures::{Stream, StreamExt};
use readable_readability::Readability;
use tokio::io::AsyncBufReadExt;
use tracing::{debug, warn};
use url::Url;

use self::types::{CrawledDocument, KatanaRequestResponse};

async fn crawl_url(start_url: &str) -> anyhow::Result<impl Stream<Item = KatanaRequestResponse>> {
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
        .spawn()?;

    let stdout = child.stdout.take().expect("Failed to acquire stdout");
    let mut stdout = tokio::io::BufReader::new(stdout).lines();

    let stderr = child.stderr.take().expect("Failed to acquire stderr");
    let mut stderr = tokio::io::BufReader::new(stderr).lines();

    tokio::spawn(async move {
        if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
            if exit_code != 0 {
                warn!("Katana exited with code {}", exit_code);
            }
        }
    });

    tokio::spawn(async move {
        while let Ok(Some(line)) = stderr.next_line().await {
            logkit::info!("{line}");
        }
    });

    Ok(stream! {
        while let Ok(Some(line)) = stdout.next_line().await {
            let data = match serde_json::from_str::<KatanaRequestResponse>(&line) {
                Ok(data) => data,
                Err(err) => {
                    warn!("Failed to parse katana output: {:?}, skipping...", err);
                    continue;
                }
            };

            if data.response.status_code == Some(429) {
                logkit::warn!("429 Too Many Requests, consider adjust your rate limit settings...");
            }

            // Skip if the status code is not 200
            if data.response.status_code != Some(200) {
                continue;
            }

            // Skip if the content type is not text/html
            if !data
                .response
                .headers
                .get("content_type")
                .is_some_and(|ct| ct.starts_with("text/html"))
            {
                continue;
            }

            // Skip if the content is larger than 1M.
            if data.response.raw.as_ref().is_some_and(|x| x.len() > 1_000_000) {
                debug!("Skipping {} as the content is larger than 1M", data.request.endpoint);
                continue;
            }

            yield data;
        }
    })
}

fn to_document(data: KatanaRequestResponse) -> Option<CrawledDocument> {
    // Cleanup the HTML with Readability
    let (html, metadata) = {
        let (node, metadata) = Readability::new()
            .base_url(Url::parse(&data.request.endpoint).ok()?)
            .parse(&data.response.body?);

        let mut html_bytes = vec![];
        node.serialize(&mut html_bytes).ok()?;
        (String::from_utf8(html_bytes).ok()?, metadata)
    };

    // Convert the HTML to Markdown
    let md = match htmd::HtmlToMarkdown::new().convert(&html) {
        Ok(md) => md,
        Err(err) => {
            warn!("Failed to convert HTML to Markdown: {:?}", err);
            return None;
        }
    };

    // Skip if the document is empty
    if md.is_empty() {
        return None;
    }

    Some(CrawledDocument::new(
        data.request.endpoint,
        md.trim().to_owned(),
        metadata.into(),
    ))
}

pub async fn crawl_pipeline(
    start_url: &str,
) -> anyhow::Result<impl Stream<Item = CrawledDocument>> {
    Ok(crawl_url(start_url)
        .await?
        .filter_map(move |data| async move { to_document(data) }))
}

#[cfg(test)]
mod tests {

    use tracing_test::traced_test;

    use super::*;

    #[tokio::test]
    #[traced_test]
    async fn test_to_document() {
        let headers = [("content_type".into(), "text/html".into())]
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
                status_code: Some(200),
                headers,
                body: Some("<p>Hello, World!</p>".to_owned()),
                technologies: Default::default(),
                raw: Some("HTTP/1.1 200 OK\nContent-Type: text/html\n".to_owned()),
            },
        };

        let doc = to_document(data).unwrap();
        assert_eq!(doc.url, "https://example.com");
        assert_eq!(doc.markdown, "Hello, World!");
    }
}
