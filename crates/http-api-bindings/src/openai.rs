use async_openai::{config::OpenAIConfig, error::OpenAIError, types::CreateCompletionRequestArgs};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tabby_inference::{CompletionOptions, CompletionStream};
use tracing::warn;

pub struct OpenAIEngine {
    client: async_openai::Client<OpenAIConfig>,
    model_name: String,
}

impl OpenAIEngine {
    pub fn create(api_endpoint: &str, model_name: &str, api_key: Option<String>) -> Self {
        let config = OpenAIConfig::default()
            .with_api_base(api_endpoint)
            .with_api_key(api_key.unwrap_or_default());

        let client = async_openai::Client::with_config(config);

        Self {
            client,
            model_name: model_name.to_owned(),
        }
    }
}

#[async_trait]
impl CompletionStream for OpenAIEngine {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        let request = CreateCompletionRequestArgs::default()
            .model(&self.model_name)
            .temperature(options.sampling_temperature)
            .stream(true)
            .prompt(prompt)
            .build();

        let s = stream! {
            let request = match request {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to build completion request {:?}", e);
                    return;
                }
            };

            let s = match self.client.completions().create_stream(request).await {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to create completion request {:?}", e);
                    return;
                }
            };

            for await x in s {
                match x {
                    Ok(x) => {
                        yield x.choices[0].text.clone();
                    },
                    Err(OpenAIError::StreamError(_)) => break,
                    Err(e) => {
                        warn!("Failed to stream response: {}", e);
                        break;
                    }
                };
            }
        };

        Box::pin(s)
    }
}
