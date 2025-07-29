use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use tabby_inference::{CompletionOptions, CompletionStream};
use tracing::warn;

use super::split_fim_prompt;

pub struct OpenAICompletionEngine {
    client: reqwest::Client,
    model_name: String,
    api_endpoint: String,
    api_key: Option<String>,

    /// OpenAI Completion API use suffix field in request when FIM is not supported,
    /// support_fim is used to mark if FIM is supported,
    /// provide a `openai/legacy_completion_no_fim` backend to use suffix field.
    support_fim: bool,
}

impl OpenAICompletionEngine {
    pub fn create(
        model_name: Option<String>,
        api_endpoint: &str,
        api_key: Option<String>,
        support_fim: bool,
    ) -> Box<dyn CompletionStream> {
        let model_name = model_name.expect("model_name is required for openai/completion");
        let client = reqwest::Client::new();

        Box::new(Self {
            client,
            model_name,
            api_endpoint: format!("{api_endpoint}/completions"),
            api_key,
            support_fim,
        })
    }
}

#[derive(Serialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    suffix: Option<String>,
    max_tokens: i32,
    temperature: f32,
    stream: bool,
    presence_penalty: f32,
}

#[derive(Deserialize)]
struct CompletionResponseChunk {
    choices: Vec<CompletionResponseChoice>,
}

#[derive(Deserialize)]
struct CompletionResponseChoice {
    text: String,
    finish_reason: Option<String>,
}

#[async_trait]
impl CompletionStream for OpenAICompletionEngine {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        let (prompt, suffix) = if self.support_fim {
            split_fim_prompt(prompt)
        } else {
            (prompt, None)
        };

        let request = CompletionRequest {
            model: self.model_name.clone(),
            prompt: prompt.to_owned(),
            suffix: suffix.map(|x| x.to_owned()),
            max_tokens: options.max_decoding_tokens,
            temperature: options.sampling_temperature,
            stream: true,
            presence_penalty: options.presence_penalty,
        };

        let mut request = self.client.post(&self.api_endpoint).json(&request);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let s = stream! {
            let mut es = EventSource::new(request).expect("Failed to create event source");
            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(message)) => {
                        let x: CompletionResponseChunk = serde_json::from_str(&message.data).expect("Failed to parse response");
                        if let Some(choice) = x.choices.first() {
                            yield choice.text.clone();

                            if choice.finish_reason.is_some() {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        match e {
                            reqwest_eventsource::Error::StreamEnded => {},
                            reqwest_eventsource::Error::InvalidStatusCode(code, resp) =>
                                warn!("Error in completion event source: {}, {}",
                                      code, resp.text().await.unwrap_or_default().replace('\n', "")),
                            _ => warn!("Error in completion event source: {}", e),
                        }
                        break;
                    }
                }
            }
        };

        Box::pin(s)
    }
}
