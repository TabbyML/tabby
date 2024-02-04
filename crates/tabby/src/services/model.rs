use std::{fs, path::PathBuf, sync::Arc};

use serde::Deserialize;
use tabby_common::{
    registry::{parse_model_id, ModelRegistry, GGML_MODEL_RELATIVE_PATH},
    terminal::{HeaderFormat, InfoMessage},
};
use tabby_download::download_model;
use tabby_inference::{make_text_generation, TextGeneration};
use tracing::info;

use crate::{fatal, Device};

pub async fn load_text_generation(
    model_id: &str,
    device: &Device,
    parallelism: u8,
) -> (Arc<dyn TextGeneration>, PromptInfo) {
    #[cfg(feature = "experimental-http")]
    if device == &Device::ExperimentalHttp {
        let (engine, prompt_template) = http_api_bindings::create(model_id);
        return (
            engine,
            PromptInfo {
                prompt_template: Some(prompt_template),
                chat_template: None,
            },
        );
    }

    if fs::metadata(model_id).is_ok() {
        let path = PathBuf::from(model_id);
        let model_path = path.join(GGML_MODEL_RELATIVE_PATH);
        let engine = create_ggml_engine(
            device,
            model_path.display().to_string().as_str(),
            parallelism,
        );
        let engine_info = PromptInfo::read(path.join("tabby.json"));
        (Arc::new(engine), engine_info)
    } else {
        let (registry, name) = parse_model_id(model_id);
        let registry = ModelRegistry::new(registry).await;
        let model_path = registry.get_model_path(name).display().to_string();
        let model_info = registry.get_model_info(name);
        let engine = create_ggml_engine(device, &model_path, parallelism);
        (
            Arc::new(engine),
            PromptInfo {
                prompt_template: model_info.prompt_template.clone(),
                chat_template: model_info.chat_template.clone(),
            },
        )
    }
}

#[derive(Deserialize)]
pub struct PromptInfo {
    pub prompt_template: Option<String>,
    pub chat_template: Option<String>,
}

impl PromptInfo {
    fn read(filepath: PathBuf) -> PromptInfo {
        serdeconv::from_json_file(&filepath)
            .unwrap_or_else(|_| fatal!("Invalid metadata file: {}", filepath.display()))
    }
}

fn create_ggml_engine(device: &Device, model_path: &str, parallelism: u8) -> impl TextGeneration {
    if !device.ggml_use_gpu() {
        InfoMessage::new(
            "CPU Device",
            HeaderFormat::BoldBlue,
            &[
                "Tabby is currently running on the CPU. Completions may be slow, but it will suffice for testing purposes.",
                "For better performance, consider deploying Tabby on a GPU device."
            ],
        );
    }
    let options = llama_cpp_bindings::LlamaTextGenerationOptionsBuilder::default()
        .model_path(model_path.to_owned())
        .use_gpu(device.ggml_use_gpu())
        .parallelism(parallelism)
        .build()
        .unwrap();

    make_text_generation(llama_cpp_bindings::LlamaTextGeneration::new(options))
}

pub async fn download_model_if_needed(model: &str) {
    if fs::metadata(model).is_ok() {
        info!("Loading model from local path {}", model);
    } else {
        download_model(model, true).await;
    }
}
