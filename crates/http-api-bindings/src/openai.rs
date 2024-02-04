use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header;
use reqwest_eventsource::{Error, Event, EventSource};
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGenerationOptions, TextGenerationStream};
use tracing::warn;

#[derive(Serialize)]
struct Request {
    model: String,
    prompt: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    stream: bool,
}

#[derive(Deserialize)]
struct Response {
    choices: Vec<Prediction>,
}

#[derive(Deserialize)]
struct Prediction {
    text: String,
}

pub struct OpenAIEngine {
    client: reqwest::Client,
    api_endpoint: String,
    model_name: String,
}

impl OpenAIEngine {
    pub fn create(api_endpoint: &str, model_name: &str, authorization: Option<String>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(authorization) = authorization {
            headers.insert(
                "Authorization",
                header::HeaderValue::from_str(&authorization)
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
}

#[async_trait]
impl TextGenerationStream for OpenAIEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> BoxStream<String> {
        let request = Request {
            model: self.model_name.to_owned(),
            prompt: vec![prompt.to_string()],
            max_tokens: options.max_decoding_length,
            temperature: options.sampling_temperature,
            stream: true,
        };

        let es = EventSource::new(self.client.post(&self.api_endpoint).json(&request));
        // API Documentation: https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md
        let s = stream! {
            let Ok(es) = es else {
                warn!("Failed to access api_endpoint: {}", &self.api_endpoint);
                return;
            };

            for await event in es {
                match event {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(message)) => {
                        let Ok(x) = serde_json::from_str::<Response>(&message.data) else {
                            warn!("Invalid response payload: {}", message.data);
                            break;
                        };
                        yield x.choices[0].text.clone();
                    }
                    Err(Error::StreamEnded) => {
                        break;
                    },
                    Err(err) => {
                        warn!("Failed to start streaming: {}", err);
                    }
                };
            }
        };

        Box::pin(s)
    }
}
