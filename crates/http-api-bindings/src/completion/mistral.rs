use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use tabby_inference::{CompletionOptions, CompletionStream};

use super::split_fim_prompt;

pub struct MistralFIMEngine {
    client: reqwest::Client,
    api_endpoint: String,
    api_key: String,
    model_name: String,
}

const DEFAULT_API_ENDPOINT: &str = "https://api.mistral.ai";

impl MistralFIMEngine {
    pub fn create(
        api_endpoint: Option<&str>,
        api_key: Option<String>,
        model_name: Option<String>,
    ) -> Box<dyn CompletionStream> {
        let client = reqwest::Client::new();
        let model_name = model_name.unwrap_or("codestral-latest".into());
        let api_key = api_key.expect("API key is required for mistral/completion");

        Box::new(Self {
            client,
            model_name,
            api_endpoint: format!(
                "{}/v1/fim/completions",
                api_endpoint.unwrap_or(DEFAULT_API_ENDPOINT)
            ),
            api_key,
        })
    }
}

#[derive(Serialize)]
struct FIMRequest {
    prompt: String,
    suffix: Option<String>,
    model: String,
    temperature: f32,
    max_tokens: i32,
    stream: bool,
    random_seed: u64,
}

#[derive(Deserialize)]
struct FIMResponseChunk {
    choices: Vec<FIMResponseChoice>,
}

#[derive(Deserialize)]
struct FIMResponseChoice {
    delta: FIMResponseDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct FIMResponseDelta {
    content: String,
}

#[async_trait]
impl CompletionStream for MistralFIMEngine {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        let (prompt, suffix) = split_fim_prompt(prompt);
        let request = FIMRequest {
            prompt: prompt.to_owned(),
            suffix: suffix.map(|x| x.to_owned()),
            model: self.model_name.clone(),
            max_tokens: options.max_decoding_tokens,
            temperature: options.sampling_temperature,
            stream: true,
            random_seed: options.seed,
        };

        let request = self
            .client
            .post(&self.api_endpoint)
            .bearer_auth(&self.api_key)
            .json(&request);

        let s = stream! {
            let mut es = EventSource::new(request).expect("Failed to create event source");
            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(message)) => {
                        let x: FIMResponseChunk = serde_json::from_str(&message.data).expect("Failed to parse response");
                        if let Some(choice) = x.choices.first() {
                            yield choice.delta.content.clone();

                            if choice.finish_reason.is_some() {
                                break;
                            }
                        }
                    }
                    Err(_) => {
                        // StreamEnd
                        break;
                    }
                }
            }
        };

        Box::pin(s)
    }
}
