//! Lays out the abstract definition of a text generation model, and utilities for encodings.
mod chat;
mod code;
mod completion;
mod decoding;
mod embedding;

pub use chat::{ChatCompletionStream, ExtendedOpenAIConfig};
pub use code::{CodeGeneration, CodeGenerationOptions, CodeGenerationOptionsBuilder};
pub use completion::{CompletionOptions, CompletionOptionsBuilder, CompletionStream};
pub use embedding::{tests::MockEmbedding, Embedding};

fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|x| x.as_millis() as u64)
        .unwrap_or_default()
}
