mod huggingface;
mod modelscope;

use anyhow::Result;
use async_trait::async_trait;
use huggingface::HuggingFaceRegistry;

use self::modelscope::ModelScopeRegistry;

#[async_trait]
pub trait Registry {
    fn build_url(&self, model_id: &str, path: &str) -> String;
    async fn build_cache_key(&self, url: &str) -> Result<String>;
}

pub fn create_registry() -> Box<dyn Registry> {
    let registry = std::env::var("TABBY_REGISTRY").unwrap_or("huggingface".to_owned());
    if registry == "huggingface" {
        Box::<HuggingFaceRegistry>::default()
    } else if registry == "modelscope" {
        Box::<ModelScopeRegistry>::default()
    } else {
        panic!("Unsupported registry {}", registry);
    }
}
