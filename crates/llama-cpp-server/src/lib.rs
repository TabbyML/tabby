use std::{process::Stdio, sync::Arc};

use serde_json::json;
use tabby_inference::{ChatCompletionStream, CompletionStream, Embedding};

struct LlamaCppServer {
    process: tokio::process::Child,
}

const SERVER_PORT: u16 = 30888;

impl LlamaCppServer {
    pub fn new(model_path: &str, use_gpu: bool, parallelism: u8) -> Self {
        let mut num_gpu_layers = std::env::var("LLAMA_CPP_N_GPU_LAYERS")
            .unwrap_or("9999".into());
        if !use_gpu {
            num_gpu_layers = "0".to_string();
        }
        let process = tokio::process::Command::new("llama-cpp-server")
            .arg("-m")
            .arg(model_path)
            .arg("--port")
            .arg(SERVER_PORT.to_string())
            .arg("-ngl")
            .arg(num_gpu_layers)
            .arg("-np")
            .arg(parallelism.to_string())
            .kill_on_drop(true)
            .spawn()
            .expect("Failed to spawn llama-cpp-server");

        Self { process }
    }

    pub fn completion(&self, prompt_template: String) -> Arc<dyn CompletionStream> {
        let model_spec: String = serde_json::to_string(&json!({
            "kind": "llama",
            "api_endpoint": format!("http://localhost:{SERVER_PORT}"),
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
