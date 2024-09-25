use axum::Json;
use tabby_common::api::server_setting::ServerSetting;

#[utoipa::path(
    get,
    path = "/v1beta/server_setting",
    tag = "v1beta",
    operation_id = "config",
    responses(
        (status = 200, description = "Success", body = ServerSetting, content_type = "application/json"),
    ),
    security(
        ("token" = [])
    )
)]
pub async fn setting() -> Json<ServerSetting> {
    let config = ServerSetting {
        disable_client_side_telemetry: false,
    };
    Json(config)
}
