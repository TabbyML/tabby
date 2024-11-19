use async_trait::async_trait;

#[async_trait]
pub trait Embedding: Sync + Send {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>>;
}

pub mod tests {
    use super::*;
    use anyhow::Result;

    pub struct MockEmbedding {
        result: Vec<f32>,
    }

    impl MockEmbedding {
        pub fn new(result: Vec<f32>) -> Self {
            Self { result }
        }
    }

    #[async_trait]
    impl Embedding for MockEmbedding {
        async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
            if prompt.starts_with("error") {
                Err(anyhow::anyhow!(prompt.to_owned()))
            } else {
                Ok(self.result.clone())
            }
        }
    }
}
