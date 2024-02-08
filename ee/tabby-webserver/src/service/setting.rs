use anyhow::Result;
use async_trait::async_trait;
use tabby_db::DbConn;
use validator::Validate;

use crate::schema::setting::{
    NetworkSetting, NetworkSettingInput, SecuritySetting, SecuritySettingInput, SettingService,
};

#[async_trait]
impl SettingService for DbConn {
    async fn read_security_setting(&self) -> Result<SecuritySetting> {
        Ok(self.read_server_setting().await?.into())
    }

    async fn update_security_setting(&self, input: SecuritySettingInput) -> Result<()> {
        input.validate()?;
        let domains = if input.allowed_register_domain_list.is_empty() {
            None
        } else {
            Some(input.allowed_register_domain_list.join(","))
        };

        self.update_security_setting(domains, input.disable_client_side_telemetry)
            .await
    }

    async fn read_network_setting(&self) -> Result<NetworkSetting> {
        Ok(self.read_server_setting().await?.into())
    }

    async fn update_network_setting(&self, input: NetworkSettingInput) -> Result<()> {
        input.validate()?;
        self.update_network_setting(input.external_url).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_setting() {
        let db = DbConn::new_in_memory().await.unwrap();

        assert_eq!(
            SettingService::read_security_setting(&db).await.unwrap(),
            SecuritySetting {
                allowed_register_domain_list: vec![],
                disable_client_side_telemetry: false,
            }
        );

        SettingService::update_security_setting(
            &db,
            SecuritySettingInput {
                allowed_register_domain_list: vec!["example.com".into()],
                disable_client_side_telemetry: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            SettingService::read_security_setting(&db).await.unwrap(),
            SecuritySetting {
                allowed_register_domain_list: vec!["example.com".into()],
                disable_client_side_telemetry: true,
            }
        );
    }

    #[tokio::test]
    async fn test_network_setting() {
        let db = DbConn::new_in_memory().await.unwrap();

        assert_eq!(
            SettingService::read_network_setting(&db).await.unwrap(),
            NetworkSetting {
                external_url: "http://localhost:8080".into(),
            }
        );

        SettingService::update_network_setting(
            &db,
            NetworkSettingInput {
                external_url: "http://localhost:8081".into(),
            },
        )
        .await
        .unwrap();

        assert_eq!(
            SettingService::read_network_setting(&db).await.unwrap(),
            NetworkSetting {
                external_url: "http://localhost:8081".into(),
            }
        );
    }
}
