use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tabby_inference::{helpers, TextGenerationOptions, TextGenerationStream};

#[derive(Serialize)]
struct Request {
    instances: Vec<Instance>,
    parameters: Parameters,
}

#[derive(Serialize)]
struct Instance {
    prefix: String,
    suffix: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Parameters {
    temperature: f32,
    max_output_tokens: usize,
    stop_sequences: Vec<String>,
}

#[derive(Deserialize)]
struct Response {
    predictions: Vec<Prediction>,
}

#[derive(Deserialize)]
struct Prediction {
    content: String,
}

pub struct VertexAIEngine {
    client: reqwest::Client,
    api_endpoint: String,
}

impl VertexAIEngine {
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
        Self {
            api_endpoint: api_endpoint.to_owned(),
            client,
        }
    }

    pub fn prompt_template() -> String {
        "{prefix}<MID>{suffix}".to_owned()
    }

    // FIXME(meng): migrate to streaming implementation
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let stop_sequences = if let Some(language) = options.language {
            language
                .get_stop_words()
                .iter()
                .map(|x| x.to_string())
                // vertex supports at most 5 stop sequence.
                .take(5)
                .collect()
        } else {
            vec![]
        };

        let tokens: Vec<&str> = prompt.split("<MID>").collect();
        let request = Request {
            instances: vec![Instance {
                prefix: tokens[0].to_owned(),
                suffix: Some(tokens[1].to_owned()),
            }],
            // options.max_input_length is ignored.
            parameters: Parameters {
                temperature: options.sampling_temperature,
                // vertex supports at most 64 output tokens.
                max_output_tokens: std::cmp::min(options.max_decoding_length, 64),
                stop_sequences,
            },
        };

        // API Documentation: https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#code-completion-prompt-parameters
        let resp = self
            .client
            .post(&self.api_endpoint)
            .json(&request)
            .send()
            .await
            .expect("Failed to making completion request");

        if resp.status() != 200 {
            let err: Value = resp.json().await.expect("Failed to parse response");
            println!("Request failed: {}", err);
            std::process::exit(1);
        }

        let resp: Response = resp.json().await.expect("Failed to parse response");

        resp.predictions[0].content.clone()
    }
}

#[async_trait]
impl TextGenerationStream for VertexAIEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> BoxStream<String> {
        helpers::string_to_stream(self.generate(prompt, options).await).await
    }
}
