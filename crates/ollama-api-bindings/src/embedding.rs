use async_trait::async_trait;
use ollama_rs::{generation::embeddings::request::GenerateEmbeddingsRequest, Ollama};
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use crate::model::{create_connection, OllamaModelExt};

pub struct OllamaCompletion {
    /// Connection to Ollama API
    connection: Ollama,
    /// Model name, <model>
    model: String,
}

#[async_trait]
impl Embedding for OllamaCompletion {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let request =
            GenerateEmbeddingsRequest::new(self.model.to_owned(), prompt.to_owned().into());

        let response = self.connection.generate_embeddings(request).await?;

        response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding data found"))
    }
}

pub async fn create(config: &HttpModelConfig) -> Box<dyn Embedding> {
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
    async fn forwards_api_key_to_model_selection_and_embedding_requests() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            for (path, body) in [
                ("/api/show", "{}"),
                ("/api/embed", "{\"embeddings\":[[1.0,2.0]]}"),
            ] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0; 4096];
                let length = socket.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..length]);
                assert!(request.starts_with(&format!("POST {path} HTTP/1.1")));
                assert!(request
                    .lines()
                    .any(|line| line.eq_ignore_ascii_case("authorization: Bearer test-key")));

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let config = HttpModelConfig {
            kind: "ollama/embedding".to_owned(),
            api_endpoint: Some(format!("http://{address}")),
            api_key: Some("test-key".to_owned()),
            rate_limit: Default::default(),
            model_name: Some("test-model".to_owned()),
            prompt_template: None,
            chat_template: None,
            supported_models: None,
            additional_stop_words: None,
        };

        let embedding = create(&config).await;
        assert_eq!(
            embedding.embed("test prompt").await.unwrap(),
            vec![1.0, 2.0]
        );
        server.await.unwrap();
    }
}
