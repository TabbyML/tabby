use axum::Json;
use tabby_common::config::Config;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct SupportedModel {
    completion: Vec<String>,
    chat: Vec<String>,
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
pub async fn models() -> Json<SupportedModel> {
    let models: tabby_common::config::ModelConfigGroup = Config::load().expect("Config file should be exist").model;
    let mut http_model_configs: SupportedModel = SupportedModel {
        completion: Vec::new(),
        chat: Vec::new(),
    };


    if let Some(tabby_common::config::ModelConfig::Http(completion_http_config)) = models.completion {
        if let Some(not_none_supported_models) = completion_http_config.supported_models {
            http_model_configs.completion.extend(not_none_supported_models);
        }
    }

    if let Some(tabby_common::config::ModelConfig::Http(http_config)) = models.chat {
        if let Some(not_none_supported_models) = http_config.supported_models {
            http_model_configs.chat.extend(not_none_supported_models);
        }
    }

    Json(http_model_configs)
}