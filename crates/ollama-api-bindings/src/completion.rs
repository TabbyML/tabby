use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use ollama_rs::{
    generation::{completion::request::GenerationRequest, options::GenerationOptions},
    Ollama,
};
use tabby_common::config::HttpModelConfig;
use tabby_inference::{CompletionOptions, CompletionStream};
use tracing::error;

use crate::model::OllamaModelExt;

pub struct OllamaCompletion {
    /// Connection to Ollama API
    connection: Ollama,
    /// Model name, <model>
    model: String,
}

#[async_trait]
impl CompletionStream for OllamaCompletion {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        // FIXME: options.presence_penalty is not used
        let ollama_options = GenerationOptions::default()
            .num_predict(options.max_decoding_tokens)
            .seed(options.seed as i32)
            .repeat_last_n(0)
            .temperature(options.sampling_temperature);
        let request = GenerationRequest::new(self.model.to_owned(), prompt.to_owned())
            .template("{{ .Prompt }}".to_string())
            .options(ollama_options);

        // Why this function returns not Result?
        match self.connection.generate_stream(request).await {
            Ok(stream) => {
                let tabby_stream = stream! {

                    for await response in stream {
                        let parts = response.unwrap();
                        for part in parts {
                            yield part.response
                        }
                    }

                };

                tabby_stream.boxed()
            }
            Err(err) => {
                error!("Failed to generate completion: {}", err);
                futures::stream::empty().boxed()
            }
        }
    }
}

pub async fn create(config: &HttpModelConfig) -> Box<dyn CompletionStream> {
    let connection = Ollama::try_new(config.api_endpoint.as_deref().unwrap().to_owned())
        .expect("Failed to create connection to Ollama, URL invalid");

    let model = connection.select_model_or_default(config).await.unwrap();

    Box::new(OllamaCompletion { connection, model })
}
