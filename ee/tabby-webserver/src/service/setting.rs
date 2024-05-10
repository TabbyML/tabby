use async_trait::async_trait;
use tabby_db::DbConn;
use tabby_schema::{
    setting::{
        NetworkSetting, NetworkSettingInput, SecuritySetting, SecuritySettingInput, SettingService,
    },
    Result,
};

struct SettingServiceImpl {
    db: DbConn,
}

pub fn create(db: DbConn) -> impl SettingService {
    SettingServiceImpl { db }
}

#[async_trait]
impl SettingService for SettingServiceImpl {
    async fn read_security_setting(&self) -> Result<SecuritySetting> {
        Ok(self.db.read_server_setting().await?.into())
    }

    async fn update_security_setting(&self, input: SecuritySettingInput) -> Result<()> {
        let domains = if input.allowed_register_domain_list.is_empty() {
            None
        } else {
            Some(input.allowed_register_domain_list.join(","))
        };

        self.db
            .update_security_setting(domains, input.disable_client_side_telemetry)
            .await?;
        Ok(())
    }

    async fn read_network_setting(&self) -> Result<NetworkSetting> {
        Ok(self.db.read_server_setting().await?.into())
    }

    async fn update_network_setting(&self, input: NetworkSettingInput) -> Result<()> {
        self.db.update_network_setting(input.external_url).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_setting() {
        let db = DbConn::new_in_memory().await.unwrap();
        let svc = create(db.clone());

        assert_eq!(
            svc.read_security_setting().await.unwrap(),
            SecuritySetting {
                allowed_register_domain_list: vec![],
                disable_client_side_telemetry: false,
            }
        );

        svc.update_security_setting(SecuritySettingInput {
            allowed_register_domain_list: vec!["example.com".into()],
            disable_client_side_telemetry: true,
        })
        .await
        .unwrap();

        assert_eq!(
            svc.read_security_setting().await.unwrap(),
            SecuritySetting {
                allowed_register_domain_list: vec!["example.com".into()],
                disable_client_side_telemetry: true,
            }
        );
    }

    #[tokio::test]
    async fn test_network_setting() {
        let db = DbConn::new_in_memory().await.unwrap();
        let svc = create(db.clone());

        assert_eq!(
            svc.read_network_setting().await.unwrap(),
            NetworkSetting {
                external_url: "http://localhost:8080".into(),
            }
        );

        svc.update_network_setting(NetworkSettingInput {
            external_url: "http://localhost:8081".into(),
        })
        .await
        .unwrap();

        assert_eq!(
            svc.read_network_setting().await.unwrap(),
            NetworkSetting {
                external_url: "http://localhost:8081".into(),
            }
        );
    }
}
