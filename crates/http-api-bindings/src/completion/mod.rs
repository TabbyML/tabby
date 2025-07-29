mod llama;
mod mistral;
mod openai;

use std::sync::Arc;

use llama::LlamaCppEngine;
use mistral::MistralFIMEngine;
use openai::OpenAICompletionEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::CompletionStream;

use super::rate_limit;

pub async fn create(model: &HttpModelConfig) -> Arc<dyn CompletionStream> {
    let engine = match model.kind.as_str() {
        "llama.cpp/completion" => LlamaCppEngine::create(
            model
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required"),
            model.api_key.clone(),
        ),
        "ollama/completion" => ollama_api_bindings::create_completion(model).await,
        "mistral/completion" => MistralFIMEngine::create(
            model.api_endpoint.as_deref(),
            model.api_key.clone(),
            model.model_name.clone(),
        ),
        x if OPENAI_LEGACY_COMPLETION_FIM_ALIASES.contains(&x) => OpenAICompletionEngine::create(
            model.model_name.clone(),
            model
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required"),
            model.api_key.clone(),
            true,
        ),
        "openai/legacy_completion_no_fim" | "vllm/completion" => OpenAICompletionEngine::create(
            model.model_name.clone(),
            model
                .api_endpoint
                .as_deref()
                .expect("api_endpoint is required"),
            model.api_key.clone(),
            false,
        ),
        unsupported_kind => {
            panic!("Unsupported model kind for http completion: {unsupported_kind}")
        }
    };

    Arc::new(rate_limit::new_completion(
        engine,
        model.rate_limit.request_per_minute,
    ))
}

const FIM_TOKEN: &str = "<|FIM|>";
const FIM_TEMPLATE: &str = "{prefix}<|FIM|>{suffix}";
const OPENAI_LEGACY_COMPLETION_FIM_ALIASES: [&str; 3] = [
    "openai/legacy_completion",
    "openai/completion",
    "deepseek/completion",
];

pub fn build_completion_prompt(model: &HttpModelConfig) -> (Option<String>, Option<String>) {
    match model.kind.as_str() {
        x if x == "mistral/completion" || OPENAI_LEGACY_COMPLETION_FIM_ALIASES.contains(&x) => {
            (Some(FIM_TEMPLATE.to_owned()), None)
        }
        _ => (model.prompt_template.clone(), model.chat_template.clone()),
    }
}

fn split_fim_prompt(prompt: &str) -> (&str, Option<&str>) {
    let parts = prompt.splitn(2, FIM_TOKEN).collect::<Vec<_>>();
    (parts[0], parts.get(1).copied())
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_split_fim_prompt_no_fim() {
        let no_fim = vec![
            ("prefix<|FIM|>suffix", ("prefix", Some("suffix"))),
            ("prefix<|FIM|>", ("prefix", Some(""))),
            ("<|FIM|>suffix", ("", Some("suffix"))),
            ("<|FIM|>", ("", Some(""))),
            ("prefix", ("prefix", None)),
        ];
        for (input, expected) in no_fim {
            assert_eq!(split_fim_prompt(input), expected);
        }
    }
}
