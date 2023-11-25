use crate::languages::{get_language, Language, EMPTY_LANGUAGE};

pub struct TextGenerationOptions {
    pub max_input_length: usize,
    pub max_decoding_length: usize,
    pub sampling_temperature: f32,
    pub language: &'static Language,
}

pub struct SnippetCollectionOptions {
    pub max_snippets_to_fetch: usize,
    pub max_snippets_chars_in_prompt: usize,
    pub max_similarity_threshold: f32,
}

impl TextGenerationOptions {
    pub fn for_chat() -> TextGenerationOptions {
        TextGenerationOptions {
            max_input_length: 2048,
            max_decoding_length: 1920,
            sampling_temperature: 0.1,
            language: &EMPTY_LANGUAGE,
        }
    }

    pub fn for_completion(language: &str) -> TextGenerationOptions {
        TextGenerationOptions {
            max_input_length: 1024 + 512,
            max_decoding_length: 128,
            sampling_temperature: 0.1,
            language: get_language(language),
        }
    }
}

impl SnippetCollectionOptions {
    pub fn default() -> Self {
        SnippetCollectionOptions {
            max_snippets_to_fetch: 20,
            max_snippets_chars_in_prompt: 768,
            max_similarity_threshold: 0.9,
        }
    }
}
