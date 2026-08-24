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

use crate::model::{create_connection, OllamaModelExt};

pub struct OllamaCompletion {
    /// Connection to Ollama API
    connection: Ollama,
    /// Model name, <model>
    model: String,
}

#[async_trait]
impl CompletionStream for OllamaCompletion {
    async fn generate(
        &self,
        prompt: &str,
        options: CompletionOptions,
    ) -> BoxStream<'life0, String> {
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
    let connection = create_connection(config);

    let model = connection.select_model_or_default(config).await.unwrap();

    Box::new(OllamaCompletion { connection, model })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn forwards_api_key_to_ollama_requests() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0; 4096];
            let length = socket.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..length]);
            assert!(request
                .lines()
                .any(|line| line.eq_ignore_ascii_case("authorization: Bearer test-key")));

            socket
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
        });

        let config = HttpModelConfig {
            kind: "ollama/completion".to_owned(),
            api_endpoint: Some(format!("http://{address}")),
            api_key: Some("test-key".to_owned()),
            rate_limit: Default::default(),
            model_name: Some("test-model".to_owned()),
            prompt_template: None,
            chat_template: None,
            supported_models: None,
            additional_stop_words: None,
        };

        let connection = create_connection(&config);
        assert!(connection.list_local_models().await.is_err());
        server.await.unwrap();
    }

    #[test]
    fn ignores_invalid_api_key() {
        let config = HttpModelConfig {
            kind: "ollama/completion".to_owned(),
            api_endpoint: Some("http://127.0.0.1:11434".to_owned()),
            api_key: Some("invalid\n-key".to_owned()),
            rate_limit: Default::default(),
            model_name: Some("test-model".to_owned()),
            prompt_template: None,
            chat_template: None,
            supported_models: None,
            additional_stop_words: None,
        };

        let _connection = create_connection(&config);
    }
}
