use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use tabby_inference::{code::NEXT_EDIT_SUGGESTION_MODE, CompletionOptions, CompletionStream};

use crate::create_reqwest_client;

pub struct LlamaCppEngine {
    client: reqwest::Client,
    api_endpoint: String,
    api_key: Option<String>,
}

impl LlamaCppEngine {
    pub fn create(api_endpoint: &str, api_key: Option<String>) -> Box<dyn CompletionStream> {
        let client = create_reqwest_client(api_endpoint);

        Box::new(Self {
            client,
            api_endpoint: format!("{}/completion", api_endpoint),
            api_key,
        })
    }
}

#[derive(Serialize)]
struct CompletionRequest {
    seed: u64,
    prompt: String,
    n_predict: i32,
    temperature: f32,
    stream: bool,
    penalty_last_n: i32,
    presence_penalty: f32,
}

#[derive(Deserialize)]
struct CompletionResponseChunk {
    content: String,
    stop: bool,
}

#[async_trait]
impl CompletionStream for LlamaCppEngine {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        let use_stream = !(options.mode == NEXT_EDIT_SUGGESTION_MODE);

        let request_body = CompletionRequest {
            seed: options.seed,
            prompt: prompt.to_owned(),
            n_predict: options.max_decoding_tokens,
            temperature: options.sampling_temperature,
            stream: use_stream,
            penalty_last_n: 0,
            presence_penalty: options.presence_penalty,
        };

        tracing::info!("Request URL: {}", &self.api_endpoint);
        let request_json = serde_json::to_string_pretty(&request_body)
            .unwrap_or_else(|_| "Failed to serialize request".to_string());
        tracing::info!("Request body: \n{}", request_json);

        let mut request = self.client.post(&self.api_endpoint).json(&request_body);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        if use_stream {
            let s = stream! {
                let mut es = EventSource::new(request).expect("Failed to create event source");
                while let Some(event) = es.next().await {
                    match event {
                        Ok(Event::Open) => {}
                        Ok(Event::Message(message)) => {
                            let x: CompletionResponseChunk = serde_json::from_str(&message.data).unwrap();
                            yield x.content.clone();
                            if x.stop {
                                break;
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
        } else {
            let s = stream! {
                tracing::info!("Using non-streaming mode for next_edit_suggestion");

                match self.client.post(&self.api_endpoint)
                    .json(&request_body)
                    .send()
                    .await {
                        Ok(response) => {
                            if !response.status().is_success() {
                                tracing::error!("Request failed with status: {}", response.status());
                                yield String::new();
                            } else {
                                let response_text = match response.text().await {
                                    Ok(text) => text,
                                    Err(e) => {
                                        tracing::error!("Failed to get response text: {}", e);
                                        yield String::new();
                                        return;
                                    }
                                };

                                tracing::info!("Received raw response: {}", response_text);

                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response_text) {
                                    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                                        if let Some(choice) = choices.first() {
                                            if let Some(text) = choice.get("text").and_then(|t| t.as_str()) {
                                                tracing::info!("Extracted text from JSON choices: {}", text);
                                                yield text.to_string();
                                                return;
                                            }
                                        }
                                    }

                                    if let Some(content) = json.get("content").and_then(|c| c.as_str()) {
                                        tracing::info!("Extracted content from JSON: {}", content);
                                        yield content.to_string();
                                        return;
                                    }

                                    tracing::error!("No recognized content field in JSON: {:?}", json);
                                    yield String::new();
                                } else {
                                    tracing::error!("Response is not valid JSON: {}", response_text);
                                    yield String::new();
                                }
                            }
                        },
                        Err(e) => {
                            tracing::error!("Request failed: {}", e);
                            yield String::new();
                        }
                    }
            };

            Box::pin(s)
        }
    }
}
