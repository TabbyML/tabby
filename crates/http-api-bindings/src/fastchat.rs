use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tabby_inference::{helpers, TextGeneration, TextGenerationOptions};

#[derive(Serialize)]
struct Request {
    model: String,
    prompt: Vec<String>,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Deserialize)]
struct Response {
    choices: Vec<Prediction>,
}

#[derive(Deserialize)]
struct Prediction {
    text: String,
}

pub struct FastChatEngine {
    client: reqwest::Client,
    api_endpoint: String,
    model_name: String,
}

impl FastChatEngine {
    pub fn create(api_endpoint: &str, model_name: &str, authorization: &str) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if !authorization.is_empty() {
            headers.insert(
                "Authorization",
                header::HeaderValue::from_str(authorization)
                    .expect("Failed to create authorization header"),
            );
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .expect("Failed to construct HTTP client");
        Self {
            api_endpoint: api_endpoint.to_owned(),
            model_name: model_name.to_owned(),
            client,
        }
    }

    pub fn prompt_template() -> String {
        "{prefix}<MID>{suffix}".to_owned()
    }
}

#[async_trait]
impl TextGeneration for FastChatEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let tokens: Vec<&str> = prompt.split("<MID>").collect();
        let request = Request {
            model: self.model_name.to_owned(),
            prompt: vec![tokens[0].to_owned()],
            max_tokens: options.max_decoding_length,
            temperature: options.sampling_temperature,
        };

        // API Documentation: https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md
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

        resp.choices[0].text.clone()
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        helpers::string_to_stream(self.generate(prompt, options).await).await
    }
}
