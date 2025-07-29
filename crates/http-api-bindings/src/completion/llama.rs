use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use tabby_inference::{CompletionOptions, CompletionStream};

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
            api_endpoint: format!("{api_endpoint}/completion"),
            api_key,
        })
    }
}

#[derive(Serialize, Debug)]
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
        // Always use streaming mode in generate method
        let request_body = CompletionRequest {
            seed: options.seed,
            prompt: prompt.to_owned(),
            n_predict: options.max_decoding_tokens,
            temperature: options.sampling_temperature,
            stream: true,
            penalty_last_n: 0,
            presence_penalty: options.presence_penalty,
        };

        let mut request = self.client.post(&self.api_endpoint).json(&request_body);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

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
    }
}
