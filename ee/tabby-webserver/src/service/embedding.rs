use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tabby_common::config::EmbeddingConfig;
use tabby_inference::Embedding;

pub struct EmbeddingServiceImpl {
    config: EmbeddingConfig,
    embedding: Arc<dyn Embedding>,
}

pub fn create(config: &EmbeddingConfig, embedding: Arc<dyn Embedding>) -> Arc<dyn Embedding> {
    Arc::new(EmbeddingServiceImpl {
        config: config.clone(),
        embedding,
    })
}

#[async_trait]
impl Embedding for EmbeddingServiceImpl {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        if self.config.max_input_length != 0 && prompt.len() > self.config.max_input_length {
            self.embedding
                .embed(clip_prompt(prompt, self.config.max_input_length))
                .await
        } else {
            self.embedding.embed(prompt).await
        }
    }
}

/// Clip the prompt to a maximum length, ensuring that the string is valid UTF-8.
/// This is necessary because the prompt may be split in the middle of a multi-byte character
/// which would cause an panic.
fn clip_prompt(s: &str, max_length: usize) -> &str {
    if s.len() <= max_length {
        return s;
    }

    let mut end = max_length;
    while !s.is_char_boundary(end) {
        end -= 1;
    }

    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clip_prompt() {
        assert_eq!(clip_prompt("hello", 5), "hello");
        assert_eq!(clip_prompt("hello", 3), "hel");

        // assert_eq!("é".as_bytes().len(), 2); // Latin-1 Supplement has length 2
        assert_eq!(clip_prompt("1é2", 1), "1");
        assert_eq!(clip_prompt("1é2", 2), "1");
        assert_eq!(clip_prompt("1é2", 3), "1é");

        // assert_eq!("世".as_bytes().len(), 3); // CJK has length 3
        assert_eq!(clip_prompt("1世2", 1), "1");
        assert_eq!(clip_prompt("1世2", 2), "1");
        assert_eq!(clip_prompt("1世2", 3), "1");
        assert_eq!(clip_prompt("1世2", 4), "1世");

        // assert_eq!("😀".as_bytes().len(), 4); // Emoji has length 4
        assert_eq!(clip_prompt("1😀2", 1), "1");
        assert_eq!(clip_prompt("1😀2", 2), "1");
        assert_eq!(clip_prompt("1😀2", 3), "1");
        assert_eq!(clip_prompt("1😀2", 4), "1");
        assert_eq!(clip_prompt("1😀2", 5), "1😀");
    }
}
