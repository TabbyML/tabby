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
const CONFIGURABLE_OPENAI_COMPLETION_ALIASES: [&str; 2] =
    ["openai/legacy_completion", "openai/completion"];

fn configured_prompt_template(model: &HttpModelConfig) -> Option<&str> {
    model
        .prompt_template
        .as_deref()
        .filter(|template| !template.trim().is_empty())
}

pub fn build_completion_prompt(model: &HttpModelConfig) -> (Option<String>, Option<String>) {
    match model.kind.as_str() {
        x if CONFIGURABLE_OPENAI_COMPLETION_ALIASES.contains(&x) => (
            Some(
                configured_prompt_template(model)
                    .unwrap_or(FIM_TEMPLATE)
                    .to_owned(),
            ),
            None,
        ),
        x if x == "mistral/completion" || x == "deepseek/completion" => {
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
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::TcpListener,
        sync::mpsc,
        thread,
        time::Duration,
        vec,
    };

    use futures::StreamExt;
    use serde_json::{json, Value};
    use tabby_inference::CompletionOptionsBuilder;

    use super::*;

    const CUSTOM_FIM_TEMPLATE: &str = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>";

    fn http_model(
        kind: &str,
        prompt_template: Option<&str>,
        api_endpoint: Option<String>,
    ) -> HttpModelConfig {
        HttpModelConfig {
            kind: kind.to_owned(),
            api_endpoint,
            api_key: None,
            rate_limit: Default::default(),
            model_name: Some("test-model".to_owned()),
            prompt_template: prompt_template.map(str::to_owned),
            chat_template: None,
            supported_models: None,
            additional_stop_words: None,
        }
    }

    async fn capture_completion_request(prompt_template: Option<&str>) -> (String, Value) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();

            let (request_line, body) = {
                let mut reader = BufReader::new(&stream);
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();

                let mut content_length = None;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header == "\r\n" {
                        break;
                    }

                    if let Some((name, value)) = header.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            content_length = Some(value.trim().parse::<usize>().unwrap());
                        }
                    }
                }

                let mut body = vec![0; content_length.expect("request must have content-length")];
                reader.read_exact(&mut body).unwrap();
                (request_line.trim_end().to_owned(), body)
            };

            sender
                .send((request_line, serde_json::from_slice(&body).unwrap()))
                .unwrap();

            let response_body =
                "data: {\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}]}\n\n";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            )
            .unwrap();
            stream.flush().unwrap();
        });

        let model = http_model(
            "openai/completion",
            prompt_template,
            Some(format!("http://{address}/v1")),
        );
        let (template, _) = build_completion_prompt(&model);
        let prompt = template
            .expect("openai/completion must have a prompt template")
            .replace("{prefix}", "prefix")
            .replace("{suffix}", "suffix");
        let engine = create(&model).await;
        let options = CompletionOptionsBuilder::default()
            .max_decoding_tokens(17)
            .sampling_temperature(0.25)
            .seed(42)
            .build()
            .unwrap();
        let mut response = engine.generate(&prompt, options).await;
        assert_eq!(response.next().await.as_deref(), Some(""));

        let request = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        server.join().unwrap();
        request
    }

    #[test]
    fn test_build_completion_prompt_prefers_configured_template_for_openai_aliases() {
        for kind in ["openai/legacy_completion", "openai/completion"] {
            for (configured, expected, split_fim_prompt) in [
                (Some(CUSTOM_FIM_TEMPLATE), CUSTOM_FIM_TEMPLATE, false),
                (None, FIM_TEMPLATE, true),
                (Some(""), FIM_TEMPLATE, true),
                (Some("   "), FIM_TEMPLATE, true),
            ] {
                let model = http_model(kind, configured, None);
                let (prompt_template, chat_template) = build_completion_prompt(&model);

                assert_eq!(
                    prompt_template.as_deref(),
                    Some(expected),
                    "kind={kind}, configured={configured:?}"
                );
                assert_eq!(chat_template, None);
                assert_eq!(
                    configured_prompt_template(&model).is_none(),
                    split_fim_prompt
                );
            }
        }
    }

    #[test]
    fn test_build_completion_prompt_keeps_deepseek_fim_transport() {
        let model = http_model("deepseek/completion", Some(CUSTOM_FIM_TEMPLATE), None);

        assert_eq!(
            build_completion_prompt(&model),
            (Some(FIM_TEMPLATE.to_owned()), None)
        );
    }

    #[tokio::test]
    async fn test_openai_completion_serializes_configured_and_fallback_prompts() {
        for (configured, expected_prompt, expected_suffix) in [
            (
                Some(CUSTOM_FIM_TEMPLATE),
                "<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>",
                None,
            ),
            (None, "prefix", Some("suffix")),
            (Some("{prefix}<|FIM|>{suffix}"), "prefix", Some("suffix")),
        ] {
            let (request_line, body) = capture_completion_request(configured).await;

            assert_eq!(request_line, "POST /v1/completions HTTP/1.1");
            assert_eq!(
                body,
                json!({
                    "model": "test-model",
                    "prompt": expected_prompt,
                    "suffix": expected_suffix,
                    "max_tokens": 17,
                    "temperature": 0.25,
                    "stream": true,
                    "presence_penalty": 0.0,
                }),
                "configured={configured:?}"
            );
        }
    }

    #[test]
    fn test_build_completion_prompt_keeps_existing_vllm_behavior() {
        for (configured, expected) in [
            (Some(CUSTOM_FIM_TEMPLATE), Some(CUSTOM_FIM_TEMPLATE)),
            (None, None),
        ] {
            let model = http_model("vllm/completion", configured, None);
            let (prompt_template, chat_template) = build_completion_prompt(&model);

            assert_eq!(
                prompt_template.as_deref(),
                expected,
                "configured={configured:?}"
            );
            assert_eq!(chat_template, None);
        }
    }

    #[test]
    fn test_build_completion_prompt_keeps_mistral_fim_template() {
        let model = http_model("mistral/completion", Some(CUSTOM_FIM_TEMPLATE), None);

        assert_eq!(
            build_completion_prompt(&model),
            (Some(FIM_TEMPLATE.to_owned()), None)
        );
    }

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
