use async_trait::async_trait;

#[async_trait]
pub trait Embedding: Sync + Send {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>>;
}
