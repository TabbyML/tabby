use axum::Json;
use tabby_common::api::server_setting::ServerSetting;
use tabby_common::config::Config;
use tabby_common::config::HttpModelConfig;

#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "v1beta",
    operation_id = "config",
    responses(
        (status = 200, description = "Success", body = ServerSetting, content_type = "application/json"),
    ),
    security(
        ("token" = [])
    )
)]
pub async fn models() -> Json<Vec<String>> {
    let models: tabby_common::config::ModelConfigGroup = Config::load().expect("Config file should be exist").model;
    let mut http_model_configs: Vec<String> = Vec::new();

    match models.embedding {
        tabby_common::config::ModelConfig::Http(http_config) => {
            http_model_configs.extend(http_config.supported_models.unwrap());
        }
        tabby_common::config::ModelConfig::Local(local_config) => {
            println!("Local Config Path:");
        }
    }

    if let Some(tabby_common::config::ModelConfig::Http(completion_http_config)) = models.completion {
        println!("{:?}", completion_http_config.supported_models);
        // http_model_configs.extend(completion_http_config.supported_models);

        if let Some(not_none_supported_models) = completion_http_config.supported_models {
            http_model_configs.extend(not_none_supported_models);
        }
    }

    if let Some(tabby_common::config::ModelConfig::Http(http_config)) = models.chat {
        // http_model_configs.extend(http_config.supported_models);
        println!("{:?}", http_config);

        if let Some(not_none_supported_models) = http_config.supported_models {
            http_model_configs.extend(not_none_supported_models);
        }
    }

    Json(http_model_configs)
}