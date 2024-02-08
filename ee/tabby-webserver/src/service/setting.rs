use anyhow::Result;
use async_trait::async_trait;
use tabby_db::DbConn;
use validator::Validate;

use crate::schema::setting::{ServerSetting, ServerSettingInput, SettingService};

#[async_trait]
impl SettingService for DbConn {
    async fn read_server_setting(&self) -> Result<ServerSetting> {
        let setting = self.read_server_setting().await?;
        Ok(setting.into())
    }

    async fn update_server_setting(&self, setting: ServerSetting) -> Result<()> {
        ServerSettingInput {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_server_setting() {
        let db = DbConn::new_in_memory().await.unwrap();
        let server_setting = SettingService::read_server_setting(&db).await.unwrap();

        assert_eq!(
            server_setting,
            ServerSetting {
                security_allowed_register_domain_list: vec![],
                security_disable_client_side_telemetry: false,
                network_external_url: "http://localhost:8080".into(),
            }
        );

        SettingService::update_server_setting(
            &db,
            ServerSetting {
                security_allowed_register_domain_list: vec!["example.com".into()],
                security_disable_client_side_telemetry: true,
                network_external_url: "http://localhost:9090".into(),
            },
        )
        .await
        .unwrap();

        let server_setting = SettingService::read_server_setting(&db).await.unwrap();
        assert_eq!(
            server_setting,
            ServerSetting {
                security_allowed_register_domain_list: vec!["example.com".into()],
                security_disable_client_side_telemetry: true,
                network_external_url: "http://localhost:9090".into(),
            }
        );
    }
}
