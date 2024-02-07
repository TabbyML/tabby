use anyhow::Result;
use async_trait::async_trait;
use tabby_db::{DbConn, ServerSettingDAO};
use validator::Validate;

use crate::schema::settings::{ServerSetting, ServerSettingValidation, SettingService};

impl From<ServerSettingDAO> for ServerSetting {
    fn from(value: ServerSettingDAO) -> Self {
        Self {
            security_allowed_register_domain_list: value
                .security_allowed_register_domain_list()
                .map(|s| s.to_owned())
                .collect(),
            security_disable_client_side_telemetry: value.security_disable_client_side_telemetry,
            network_external_url: value.network_external_url,
        }
    }
}

#[async_trait]
impl SettingService for DbConn {
    async fn read_server_setting(&self) -> Result<ServerSetting> {
        let setting = self.read_server_setting().await?;
        Ok(setting.into())
    }

    async fn update_server_setting(&self, setting: ServerSetting) -> Result<()> {
        ServerSettingValidation {
            security_allowed_register_domain_list: setting
                .security_allowed_register_domain_list
                .iter()
                .map(|s| &**s)
                .collect(),
            network_external_url: &setting.network_external_url,
        }
        .validate()?;
        let allowed_domains = setting.security_allowed_register_domain_list.join(",");
        let allowed_domains = (!allowed_domains.is_empty()).then_some(allowed_domains);
        self.update_server_setting(
            allowed_domains,
            setting.security_disable_client_side_telemetry,
            setting.network_external_url,
        )
        .await?;
        Ok(())
    }
}
