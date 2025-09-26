use std::sync::Arc;

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use tabby_common::api::server_setting::ServerSetting;
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

        if let Some(chat) = models.chat {
            let mut chat_models = vec![];
            for model in chat.get_http_configs().iter() {
                chat_models.push(model.model_title_and_name().0);
                if let (Some(supported_models), Some(model_name)) =
                    (&model.supported_models, &model.model_name)
                {
                    chat_models.extend(
                        supported_models
                            .iter()
                            .filter(|m| model_name != *m)
                            .cloned(),
                    );
                }
            }
            http_model_configs.chat = Some(chat_models);
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
