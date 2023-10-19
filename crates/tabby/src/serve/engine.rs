use std::path::Path;

use ctranslate2_bindings::{CTranslate2Engine, CTranslate2EngineOptionsBuilder};
use serde::Deserialize;
use tabby_common::path::ModelDir;
use tabby_inference::TextGeneration;

use crate::fatal;

pub fn create_engine(
    model: &str,
    args: &crate::serve::ServeArgs,
) -> (Box<dyn TextGeneration>, EngineInfo) {
    if args.device != super::Device::ExperimentalHttp {
        let model_dir = get_model_dir(model);
        let metadata = read_metadata(&model_dir);
        let engine = create_local_engine(args, &model_dir, &metadata);
        (
            engine,
            EngineInfo {
                prompt_template: metadata.prompt_template,
                chat_template: metadata.chat_template,
            },
        )
    } else {
        let (engine, prompt_template) = http_api_bindings::create(model);
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

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn create_local_engine(
    args: &crate::serve::ServeArgs,
    model_dir: &ModelDir,
    metadata: &Metadata,
) -> Box<dyn TextGeneration> {
    create_ctranslate2_engine(args, model_dir, metadata)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn create_local_engine(
    args: &crate::serve::ServeArgs,
    model_dir: &ModelDir,
    metadata: &Metadata,
) -> Box<dyn TextGeneration> {
    if args.device != super::Device::Metal {
        create_ctranslate2_engine(args, model_dir, metadata)
    } else {
        create_llama_engine(model_dir)
    }
}

fn create_ctranslate2_engine(
    args: &crate::serve::ServeArgs,
    model_dir: &ModelDir,
    metadata: &Metadata,
) -> Box<dyn TextGeneration> {
    let device = format!("{}", args.device);
    let compute_type = format!("{}", args.compute_type);
    let options = CTranslate2EngineOptionsBuilder::default()
        .model_path(model_dir.ctranslate2_dir())
        .tokenizer_path(model_dir.tokenizer_file())
        .device(device)
        .model_type(metadata.auto_model.clone())
        .device_indices(args.device_indices.clone())
        .num_replicas_per_device(args.num_replicas_per_device)
        .compute_type(compute_type)
        .build()
        .unwrap();
    Box::new(CTranslate2Engine::create(options))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn create_llama_engine(model_dir: &ModelDir) -> Box<dyn TextGeneration> {
    let options = llama_cpp_bindings::LlamaEngineOptionsBuilder::default()
        .model_path(model_dir.ggml_q8_0_file())
        .tokenizer_path(model_dir.tokenizer_file())
        .build()
        .unwrap();

    Box::new(llama_cpp_bindings::LlamaEngine::create(options))
}

fn get_model_dir(model: &str) -> ModelDir {
    if Path::new(model).exists() {
        ModelDir::from(model)
    } else {
        ModelDir::new(model)
    }
}

#[derive(Deserialize)]
struct Metadata {
    auto_model: String,
    prompt_template: Option<String>,
    chat_template: Option<String>,
}

fn read_metadata(model_dir: &ModelDir) -> Metadata {
    serdeconv::from_json_file(model_dir.metadata_file())
        .unwrap_or_else(|_| fatal!("Invalid metadata file: {}", model_dir.metadata_file()))
}
