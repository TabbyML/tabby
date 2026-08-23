use async_trait::async_trait;
use ollama_rs::{generation::embeddings::request::GenerateEmbeddingsRequest, Ollama};
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;

use crate::model::OllamaModelExt;

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
    let connection = Ollama::try_new(config.api_endpoint.as_deref().unwrap().to_owned())
        .expect("Failed to create connection to Ollama, URL invalid");

    let model = connection.select_model_or_default(config).await.unwrap();

    Box::new(OllamaCompletion { connection, model })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn uses_the_first_embedding_from_ollama_response() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0; 4096];
            let length = socket.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..length]);
            assert!(request.starts_with("POST /api/embed HTTP/1.1"));

            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 26\r\nConnection: close\r\n\r\n{\"embeddings\":[[1.0,2.0]]}",
                )
                .await
                .unwrap();
        });

        let connection = Ollama::try_new(format!("http://{address}")).unwrap();
        let completion = OllamaCompletion {
            connection,
            model: "test-model".to_owned(),
        };

        assert_eq!(
            completion.embed("test prompt").await.unwrap(),
            vec![1.0, 2.0]
        );
        server.await.unwrap();
    }
}
