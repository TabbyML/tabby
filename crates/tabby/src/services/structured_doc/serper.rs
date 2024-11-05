use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tabby_common::api::structured_doc::{
    DocSearch, DocSearchDocument, DocSearchError, DocSearchHit, DocSearchResponse,
    DocSearchWebDocument,
};
use tracing::warn;

#[derive(Debug, Serialize)]
struct SerperRequest {
    q: String,
    num: usize,
    page: usize,
}

#[derive(Debug, Deserialize)]
struct SerperResponse {
    organic: Vec<SerperOrganicHit>,
}

#[derive(Debug, Deserialize)]
struct SerperOrganicHit {
    title: String,
    snippet: String,
    link: String,
}

pub struct SerperService {
    client: reqwest::Client,
}

impl SerperService {
    pub fn new(api_key: &str) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "X-API-KEY",
            api_key.parse().expect("Failed to parse Serper API key"),
        );
        Self {
            client: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .expect("Failed to create reqwest client"),
        }
    }
}

#[async_trait]
impl DocSearch for SerperService {
    async fn search(
        &self,
        source_ids: &[String],
        q: &str,
        limit: usize,
    ) -> Result<DocSearchResponse, DocSearchError> {
        if !source_ids.is_empty() {
            warn!("Serper does not support source filtering");
        }

        let request = SerperRequest {
            q: q.to_string(),
            num: limit,
            page: 0,
        };
        let response = self
            .client
            .post("https://google.serper.dev/search")
            .json(&request)
            .send()
            .await
            .map_err(|e| DocSearchError::Other(e.into()))?
            .json::<SerperResponse>()
            .await
            .map_err(|e| DocSearchError::Other(e.into()))?;

        let hits = response
            .organic
            .into_iter()
            .map(|hit| DocSearchHit {
                score: 0.0,
                doc: DocSearchDocument::Web(DocSearchWebDocument {
                    title: hit.title,
                    link: hit.link,
                    snippet: hit.snippet,
                }),
            })
            .collect();

        Ok(DocSearchResponse { hits })
    }
}
