use anyhow::Result;
use async_trait::async_trait;
use juniper::GraphQLObject;

#[async_trait]
pub trait SettingService: Send + Sync {
    async fn read_server_setting(&self) -> Result<ServerSetting>;
    async fn update_server_setting(&self, setting: ServerSetting) -> Result<()>;
}

#[derive(GraphQLObject)]
pub struct ServerSetting {
    pub security_allowed_register_domain_list: Vec<String>,
    pub security_disable_client_side_telemetry: bool,
    pub network_external_url: String,
}
