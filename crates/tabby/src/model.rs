use clap::Subcommand;
use tabby_common::registry::{parse_model_id, ModelRegistry, DEFAULT_MODEL_REGISTRY_NAME};

#[derive(Subcommand)]
pub enum ModelArgs {
    #[clap(alias = "rm")]
    Delete { model: String },
    #[clap(alias = "ls")]
    List { registry: Option<String> },
    #[clap(alias = "get")]
    Download { model: String },
}

pub async fn main(args: ModelArgs) {
    match args {
        ModelArgs::Delete { model } => delete_model(&model).await,
        ModelArgs::List { registry } => list_models(registry).await,
        ModelArgs::Download { model } => download_model(&model).await,
    }
}

async fn delete_model(model: &str) {
    let (registry, name) = parse_model_id(&model);
    let registry = ModelRegistry::new(registry).await;
    let path = registry.get_model_dir(name);
    if !path.exists() {
        println!("Model {model} is not downloaded");
        return;
    }
    tokio::fs::remove_dir_all(path).await.unwrap();
    println!("Model {model} removed");
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
