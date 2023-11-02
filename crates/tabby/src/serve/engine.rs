use std::fs;

use tabby_common::registry::{parse_model_id, ModelRegistry};
use tabby_inference::TextGeneration;

pub async fn create_engine(
    model_id: &str,
    args: &crate::serve::ServeArgs,
) -> (Box<dyn TextGeneration>, EngineInfo) {
    if args.device != super::Device::ExperimentalHttp {
        if fs::metadata(model_id).is_ok() {
            let engine = create_ggml_engine(&args.device, model_id);
            (
                engine,
                EngineInfo {
                    prompt_template: args.prompt_template.clone(),
                    chat_template: args.chat_template.clone(),
                },
            )
        } else {
            let (registry, name) = parse_model_id(model_id);
            let registry = ModelRegistry::new(registry).await;
            let model_path = registry.get_model_path(name).display().to_string();
            let model_info = registry.get_model_info(name);
            let engine = create_ggml_engine(&args.device, &model_path);
            (
                engine,
                EngineInfo {
                    prompt_template: model_info.prompt_template.clone(),
                    chat_template: model_info.chat_template.clone(),
                },
            )
        }
    } else {
        let (engine, prompt_template) = http_api_bindings::create(model_id);
        (
            engine,
            EngineInfo {
                prompt_template: Some(prompt_template),
                chat_template: None,
            },
        )
    }
}

pub struct EngineInfo {
    pub prompt_template: Option<String>,
    pub chat_template: Option<String>,
}

fn create_ggml_engine(device: &super::Device, model_path: &str) -> Box<dyn TextGeneration> {
    let options = llama_cpp_bindings::LlamaTextGenerationOptionsBuilder::default()
        .model_path(model_path.to_owned())
        .use_gpu(device.ggml_use_gpu())
        .build()
        .unwrap();

    Box::new(llama_cpp_bindings::LlamaTextGeneration::create(options))
}
