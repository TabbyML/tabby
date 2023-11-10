use std::{fs, path::PathBuf};

use serde::Deserialize;
use tabby_common::registry::{parse_model_id, ModelRegistry, GGML_MODEL_RELATIVE_PATH};
use tabby_inference::TextGeneration;

use crate::fatal;

pub async fn create_engine(
    model_id: &str,
    args: &crate::serve::ServeArgs,
) -> (Box<dyn TextGeneration>, EngineInfo) {
    #[cfg(feature = "experimental-http")]
    if args.device == super::Device::ExperimentalHttp {
        let (engine, prompt_template) = http_api_bindings::create(model_id);
        return (
            engine,
            EngineInfo {
                prompt_template: Some(prompt_template),
                chat_template: None,
            },
        );
    }

    if fs::metadata(model_id).is_ok() {
        let path = PathBuf::from(model_id);
        let model_path = path.join(GGML_MODEL_RELATIVE_PATH);
        let engine = create_ggml_engine(
            &args.device,
            model_path.display().to_string().as_str(),
            args.parallelism,
        );
        let engine_info = EngineInfo::read(path.join("tabby.json"));
        (engine, engine_info)
    } else {
        let (registry, name) = parse_model_id(model_id);
        let registry = ModelRegistry::new(registry).await;
        let model_path = registry.get_model_path(name).display().to_string();
        let model_info = registry.get_model_info(name);
        let engine = create_ggml_engine(&args.device, &model_path, args.parallelism);
        (
            engine,
            EngineInfo {
                prompt_template: model_info.prompt_template.clone(),
                chat_template: model_info.chat_template.clone(),
            },
        )
    }
}

#[derive(Deserialize)]
pub struct EngineInfo {
    pub prompt_template: Option<String>,
    pub chat_template: Option<String>,
}

impl EngineInfo {
    fn read(filepath: PathBuf) -> EngineInfo {
        serdeconv::from_json_file(&filepath)
            .unwrap_or_else(|_| fatal!("Invalid metadata file: {}", filepath.display()))
    }
}

fn create_ggml_engine(
    device: &super::Device,
    model_path: &str,
    parallelism: u8,
) -> Box<dyn TextGeneration> {
    let options = llama_cpp_bindings::LlamaTextGenerationOptionsBuilder::default()
        .model_path(model_path.to_owned())
        .use_gpu(device.ggml_use_gpu())
        .parallelism(parallelism)
        .build()
        .unwrap();

    Box::new(llama_cpp_bindings::LlamaTextGeneration::new(options))
}
