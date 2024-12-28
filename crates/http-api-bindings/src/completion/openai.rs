use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use tabby_inference::{CompletionOptions, CompletionStream};
use tracing::debug;
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
            api_endpoint: format!("{}/completions", api_endpoint),
            api_key,
            support_fim,
        })
    }
}

#[derive(Debug, Default, Serialize)]
struct CompletionRequest {
    model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,

    max_tokens: i32,
    temperature: f32,
    stream: bool,
    presence_penalty: f32,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    messages: Vec<Message>,
}

#[derive(Serialize, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CompletionResponseChunk {
    choices: Vec<CompletionResponseChoice>,
}

#[derive(Deserialize, Default)]
struct CompletionResponseChoice {
    text: Option<String>,
    delta: Option<CompletionResponseDelta>,
    finish_reason: Option<String>,
}

#[derive(Clone, Deserialize, Default, Debug)]
struct CompletionResponseDelta {
    content: Option<String>,
}

#[async_trait]
impl CompletionStream for OpenAICompletionEngine {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        const LEGACY_MODEL_NAME: &str = "gpt-3.5-turbo-instruct";

        let mut request = CompletionRequest {
            model: self.model_name.clone(),
            max_tokens: options.max_decoding_tokens,
            temperature: 1_f32,
            stream: true,
            presence_penalty: options.presence_penalty,
            ..Default::default()
        };

        if self.model_name == LEGACY_MODEL_NAME {
            let (prompt, suffix) = if self.support_fim {
                split_fim_prompt(prompt)
            } else {
                (prompt, None)
            };
            request.prompt = Some(prompt.into());
            request.suffix = suffix.map(Into::into);
        } else {
            //FIXME: We may need to refine this prompt to achieve better completion results
            const SYS_PMT: &str = r#"You are a code completion assistant that completes partial code snippets. Only provide the completion portion, not the full line.
                            Examples:
                            System.out.p ->rintln("Hello");
                            Input: "String name = ->"John Doe";
                            Input: "if(num > 0->) {
                            Input: ArrayList<String> list = new -> ArrayList<>();"
                            Rules:

                            Return only the completion portion
                            Include necessary closing syntax (parentheses, quotes, semicolons)
                            Provide the most common completion for the context
                            No explanations or alternatives

                            Following is the code snippet you shouled work on:
                            "#;

            request.messages = vec![Message {
                role: "user".to_string(),
                content: format!("{SYS_PMT}\n{prompt}"),
            }];

            debug!("messages:{:?}", request.messages);
        }

        debug!("messages: {:?}", request.messages);
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

                        //message.data example: {"id":"chatcmpl-AigtaSN","object":"chat.completion.chunk","created":1735214686,"model":"o1-mini-2024-09-12","system_fingerprint":"fp_e4","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}
                        let x: CompletionResponseChunk = serde_json::from_str(&message.data).expect("Failed to parse response");
                        if let Some(choice) = x.choices.first() {
                            if choice.finish_reason.is_some() {
                                break;
                            }

                            if self.model_name ==  LEGACY_MODEL_NAME{
                                if let Some(text) = choice.text.clone(){
                                yield text;
                                }
                            }
                            else if let Some(delta) =  choice.delta.clone(){
                                debug!("delta:{:?}",delta.content);
                                yield delta.content.unwrap_or_default();
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
