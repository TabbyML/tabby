use std::sync::Arc;





use tabby_inference::{Embedding};




use super::model;
use crate::Device;

pub async fn create(model: &str, device: &Device) -> Arc<dyn Embedding> {
    model::load_embedding(model, device).await
}
