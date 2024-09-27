use std::sync::Arc;

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ModelInfo {
    completion: Option<Vec<String>>,
    chat: Option<Vec<String>>,
}

impl From<tabby_common::config::Config> for ModelInfo {
    fn from(value: tabby_common::config::Config) -> Self {
        let models = value.model;
        let mut http_model_configs: ModelInfo = ModelInfo {
            completion: None,
            chat: None,
        };

        if let Some(tabby_common::config::ModelConfig::Http(completion_http_config)) =
            models.completion
        {
            if let Some(models) = completion_http_config.supported_models {
                http_model_configs.completion = Some(models.clone());
            }
        }

        if let Some(tabby_common::config::ModelConfig::Http(chat_http_config)) = models.chat {
            if let Some(models) = chat_http_config.supported_models {
                http_model_configs.chat = Some(models.clone());
            }
        }

        http_model_configs
    }
}

#[utoipa::path(
    get,
    path = "/v1beta/models",
    tag = "v1beta",
    operation_id = "config",
    responses(
        (status = 200, description = "Success", body = ServerSetting, content_type = "application/json"),
    ),
    security(
        ("token" = [])
    )
)]
pub async fn models(State(state): State<Arc<ModelInfo>>) -> Json<ModelInfo> {
    Json(state.as_ref().clone())
}
