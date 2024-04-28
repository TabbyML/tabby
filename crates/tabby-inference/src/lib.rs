//! Lays out the abstract definition of a text generation model, and utilities for encodings.
mod chat;
mod completion;
mod decoding;
mod generation;

pub use chat::{ChatCompletionOptions, ChatCompletionOptionsBuilder, ChatCompletionStream};
pub use completion::{CompletionOptions, CompletionOptionsBuilder, CompletionStream};
pub use generation::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};

fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|x| x.as_millis() as u64)
        .unwrap_or_default()
}
