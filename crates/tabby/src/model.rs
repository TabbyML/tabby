use std::path::Path;

use clap::Subcommand;
use tabby_common::registry::{parse_model_id, ModelRegistry, DEFAULT_MODEL_REGISTRY_NAME};

#[derive(Subcommand)]
pub enum ModelArgs {
    #[clap(alias = "rm")]
    /// Delete a locally-downloaded model.
    Delete { model: String },
    #[clap(alias = "ls")]
    /// List the available models specified by a registry.
    /// This will show all models available in the given registry, and indicate which ones are installed.
    List { registry: Option<String> },
    #[clap(alias = "get", alias = "install")]
    /// Download a model from a registry.
    Download { model: String },
    /// Find the path to an installed model locally, and display its installed size in gigabytes.
    #[clap(alias = "dir", alias = "path", alias = "check")]
    Find { model: String },
}

pub async fn main(args: ModelArgs) {
    match args {
        ModelArgs::Delete { model } => delete_model(&model).await,
        ModelArgs::List { registry } => list_models(registry).await,
        ModelArgs::Download { model } => download_model(&model).await,
        ModelArgs::Find { model } => find_model(&model).await,
    }
}

async fn delete_model(model: &str) {
    let (registry, name) = parse_model_id(model);
    let registry = ModelRegistry::new(registry).await;
    let path = registry.get_model_dir(name);
    if !path.exists() {
        println!("Model {model} is not downloaded");
        return;
    }
    tokio::fs::remove_dir_all(path).await.unwrap();
    println!("Model {model} removed");
}

const GIGABYTE: f64 = (1024 * 1024 * 1024) as f64;

async fn find_model(model: &str) {
    let (registry, name) = parse_model_id(model);
    let registry = ModelRegistry::new(registry).await;
    let dir = registry.get_model_dir(name);
    if !dir.exists() {
        println!("Model is not installed");
        return;
    }
    let size = model_size(&dir);
    let size = (size as f64) / GIGABYTE;
    println!("{}: {size:.1}GB", dir.to_string_lossy());
}

fn model_size(dir: &Path) -> usize {
    let mut size = 0;
    for file in std::fs::read_dir(dir).expect("Failed to read model directory") {
        let entry = file.expect("Failed to read file metadata");
        if entry.path().is_dir() {
            size += model_size(&entry.path());
            continue;
        }
        size += entry
            .metadata()
            .expect("Failed to read file metadata")
            .len() as usize;
    }
    size
}

async fn list_models(registry: Option<String>) {
    let registry = registry.as_deref().unwrap_or(DEFAULT_MODEL_REGISTRY_NAME);
    let registry = ModelRegistry::new(registry).await;
    for model in &registry.models {
        let installed = registry
            .get_model_path(&model.name)
            .exists()
            .then_some("[Installed]")
            .unwrap_or_default();
        println!("{} {installed}", model.name);
    }
}

async fn download_model(model: &str) {
    let (registry, name) = parse_model_id(model);
    let registry = ModelRegistry::new(registry).await;
    if registry.get_model_dir(name).exists() {
        println!("Model {model} already downloaded");
        return;
    }
    tabby_download::download_model(model, true).await
}
