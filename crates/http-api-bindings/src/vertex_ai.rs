use async_trait::async_trait;
use reqwest::header;
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGeneration, TextGenerationOptions};

#[derive(Serialize)]
struct Request {
    instances: Vec<Instance>,
    parameters: Parameters
}

#[derive(Serialize)]
struct Instance {
    prefix: String,
    suffix: Option<String>
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Parameters {
    temperature: f32,
    max_output_tokens: usize,
    stop_sequences: Vec<String>
}

#[derive(Deserialize)]
struct Response {
    predictions: Vec<Prediction>
}

#[derive(Deserialize)]
struct Prediction {
    content: String
}

pub struct CodeGeckoEngine {
    client: reqwest::Client,
    api_endpoint: String,
}

impl CodeGeckoEngine {
    pub fn create(api_endpoint: &str, authorization: &str) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            header::HeaderValue::from_str(authorization)
                .expect("Failed to create authorization header"),
        );
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .expect("Failed to construct HTTP client");
        CodeGeckoEngine { api_endpoint: api_endpoint.to_owned(), client }
    }
}

#[async_trait]
impl TextGeneration for CodeGeckoEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let stop_sequences: Vec<String> = options.stop_words
            .iter()
            .map(|x| x.to_string())
            .collect();

        let request = Request {
            instances: vec![
                Instance {
                    prefix: prompt.to_owned(),
                    suffix: None,
                }
            ],
            parameters: Parameters {
                temperature: options.sampling_temperature,
                max_output_tokens: options.max_decoding_length,
                stop_sequences
            }
        };

        let resp: Response = self
            .client
            .post(&self.api_endpoint)
            .json(&request)
            .send()
            .await
            .expect("Failed to making completion request")
            .json()
            .await
            .expect("Failed to parse response");

        resp.predictions[0].content.clone()
    }
}
