mod types;

use std::process::Stdio;

use async_stream::stream;
use futures::{Stream, StreamExt};
use readable_readability::Readability;
use tokio::io::AsyncBufReadExt;
use tracing::{debug, warn};
use types::CrawledMetadata;
use url::Url;

use self::types::{CrawledDocument, KatanaRequestResponse};

async fn crawl_url(
    start_url: &str,
    prefix_url: &str,
) -> anyhow::Result<impl Stream<Item = KatanaRequestResponse>> {
    let mut command = tokio::process::Command::new("katana");

    if std::env::var("TABBY_CRAWL_ENABLE_HEADLESS").is_ok() {
        command
            .arg("-headless")
            .arg("-headless-options")
            .arg("--disable-gpu");
    }

    command
        .arg("-u")
        .arg(start_url)
        .arg("-jsonl")
        .arg("-crawl-scope")
        .arg(format!("{}.*", regex::escape(prefix_url)))
        .arg("-crawl-out-scope")
        .arg(r#"\.js$|\.css$|\.png$|\.jpg$|\.jpeg$"#)
        .arg("-depth")
        .arg("9999")
        .arg("-max-response-size")
        .arg("10485760") // 10MB max body size
        .arg("-rate-limit-minute")
        .arg("120")
        .arg("-strategy")
        .arg("breadth-first")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;

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
    prefix_url: &str,
) -> anyhow::Result<impl Stream<Item = CrawledDocument>> {
    Ok(crawl_url(start_url, prefix_url)
        .await?
        .filter_map(move |data| async move { to_document(data) }))
}

/// Attempts to fetch `llms-full.txt` from the given base URL,
/// then splits its markdown content into multiple sections based on H1 headings.
/// Each section becomes a separate `CrawledDocument`.
/// Returns a vector of `CrawledDocument`s if successful.
pub async fn crawler_llms(start_url: &str) -> anyhow::Result<Vec<CrawledDocument>> {
    // Remove trailing slash from the base URL if present.
    let base_url = start_url.trim_end_matches('/');

    let llms_full_url = format!("{}/llms-full.txt", base_url);
    let resp = reqwest::get(&llms_full_url).await?;
    if !resp.status().is_success() {
        anyhow::bail!("Unable to fetch llms-full.txt from {}", base_url);
    }
    let body = resp.text().await?;
    debug!("Successfully fetched llms-full.txt: {}", llms_full_url);

    // Split the fetched markdown content into sections.
    let docs = split_llms_content(&body, start_url);
    if docs.is_empty() {
        anyhow::bail!("No sections found in llms-full.txt from {}", base_url);
    }

    Ok(docs)
}

fn split_llms_content(content: &str, base_url: &str) -> Vec<CrawledDocument> {
    let mut docs = Vec::new();
    let mut current_title: Option<String> = None;
    let mut current_url: Option<String> = None;
    let mut current_body = String::new();

    // Process the content line by line.
    for line in content.lines() {
        // Check if the line starts with a heading-1 marker.
        if line.starts_with("# ") {
            // If we already have a section in progress, finalize it.
            if let Some(title) = current_title.take() {
                // Use the URL from the section if available; otherwise, fallback to base_url.
                let url = current_url.take().unwrap_or_else(|| base_url.to_owned());
                let metadata = CrawledMetadata {
                    title: title.into(),
                    description: url.clone().into(),
                };
                docs.push(CrawledDocument::new(
                    url,
                    current_body.trim().to_owned(),
                    metadata,
                ));
                current_body = String::new();
            }
            current_title = Some(line[2..].trim().to_owned());
            current_url = None;
        } else if line.starts_with("URL:") || line.starts_with("Source:") {
            let prefix_len = if line.starts_with("URL:") { 4 } else { 7 };
            let url_str = line[prefix_len..].trim();
            current_url = Some(url_str.to_owned());
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    // Finalize the last section if any.
    if let Some(title) = current_title {
        let url = current_url.unwrap_or_else(|| base_url.to_owned());
        let metadata = CrawledMetadata {
            title: title.into(),
            description: url.clone().into(),
        };
        docs.push(CrawledDocument::new(
            url,
            current_body.trim().to_owned(),
            metadata,
        ));
    }

    docs
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
