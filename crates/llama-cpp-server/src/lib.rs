use std::{process::Stdio, sync::Arc};

use serde_json::json;
use tabby_inference::{ChatCompletionStream, CompletionStream, Embedding};
use tokio::task::JoinHandle;
use tracing::warn;

struct LlamaCppServer {
    handle: JoinHandle<()>,
}

const SERVER_PORT: u16 = 30888;

impl LlamaCppServer {
    pub fn new(model_path: &str, use_gpu: bool, parallelism: u8) -> Self {
        let mut num_gpu_layers = std::env::var("LLAMA_CPP_N_GPU_LAYERS").unwrap_or("9999".into());
        if !use_gpu {
            num_gpu_layers = "0".to_string();
        }

        let model_path = model_path.to_owned();
        let handle = tokio::spawn(async move {
            loop {
                let mut process = tokio::process::Command::new("llama-server")
                    .arg("-m")
                    .arg(&model_path)
                    .arg("--port")
                    .arg(SERVER_PORT.to_string())
                    .arg("-ngl")
                    .arg(&num_gpu_layers)
                    .arg("-np")
                    .arg(parallelism.to_string())
                    .kill_on_drop(true)
                    .stderr(Stdio::inherit())
                    .stdout(Stdio::inherit())
                    .spawn()
                    .expect("Failed to spawn llama-cpp-server");

                let status_code = process
                    .wait()
                    .await
                    .ok()
                    .and_then(|s| s.code())
                    .unwrap_or(-1);

                if status_code != 0 {
                    warn!(
                        "llama-server exited with status code {}, restarting...",
                        status_code
                    );
                }
            }
        });

        Self { handle }
    }

    async fn wait_for_health(&self) {
        let client = reqwest::Client::new();
        loop {
            let Ok(resp) = client.get(api_endpoint() + "/health").send().await else {
                continue;
            };

            if resp.status().is_success() {
                return;
            }
        }
    }

    pub fn completion(&self, prompt_template: String) -> Arc<dyn CompletionStream> {
        let model_spec: String = serde_json::to_string(&json!({
            "kind": "llama",
            "api_endpoint": api_endpoint(),
            "prompt_template": prompt_template,
        }))
        .expect("Failed to serialize model spec");
        let (engine, _, _) = http_api_bindings::create(&model_spec);
        engine
    }

    pub fn chat(&self) -> Arc<dyn ChatCompletionStream> {
        let model_spec: String = serde_json::to_string(&json!({
            "kind": "openai-chat",
            "api_endpoint": format!("http://localhost:{SERVER_PORT}/v1"),
        }))
        .expect("Failed to serialize model spec");
        http_api_bindings::create_chat(&model_spec)
    }

    pub fn embedding(self) -> Arc<dyn Embedding> {
        let model_spec: String = serde_json::to_string(&json!({
            "kind": "llama",
            "api_endpoint": format!("http://localhost:{SERVER_PORT}"),
        }))
        .expect("Failed to serialize model spec");
        http_api_bindings::create_embedding(&model_spec)
    }
}

impl Drop for LlamaCppServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

fn api_endpoint() -> String {
    format!("http://localhost:{SERVER_PORT}")
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use tabby_common::registry::{parse_model_id, ModelRegistry};
    use tabby_inference::CompletionOptionsBuilder;

    use super::*;

    #[tokio::test]
    #[ignore = "Should only be run in local manual testing"]
    async fn test_create_completion() {
        let model_id = "StarCoder-1B";
        let (registry, name) = parse_model_id(model_id);
        let registry = ModelRegistry::new(registry).await;
        let model_path = registry.get_model_path(name).display().to_string();
        let model_info = registry.get_model_info(name);

        let server = LlamaCppServer::new(&model_path, false, 1);
        server.wait_for_health().await;

        let completion = server.completion(model_info.prompt_template.clone().unwrap());
        let s = completion
            .generate(
                "def fib(n):",
                CompletionOptionsBuilder::default()
                    .max_decoding_tokens(7)
                    .max_input_length(1024)
                    .sampling_temperature(0.0)
                    .seed(12345)
                    .build()
                    .unwrap(),
            )
            .await;

        let content: Vec<String> = s.collect().await;

        let content = content.join("");
        assert_eq!(content, "\n    if n <= 1:")
    }
}
