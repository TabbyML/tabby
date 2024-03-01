use anyhow::Result;
use sqlx::{prelude::FromRow, query, Sqlite, Transaction};

use crate::DbConn;

#[derive(Debug, PartialEq, FromRow)]
pub struct ServerSettingDAO {
    billing_enterprise_license: Option<String>,
    security_allowed_register_domain_list: Option<String>,
    pub security_disable_client_side_telemetry: bool,
    pub network_external_url: String,
}

const SERVER_SETTING_ROW_ID: i32 = 1;

impl ServerSettingDAO {
    pub fn security_allowed_register_domain_list(&self) -> impl Iterator<Item = &str> {
        self.security_allowed_register_domain_list
            .iter()
            .flat_map(|s| s.split(','))
            .map(|x| x.trim())
            .filter(|s| !s.is_empty())
    }
}

impl DbConn {
    async fn internal_read_server_setting(
        &self,
        transaction: &mut Transaction<'_, Sqlite>,
    ) -> Result<Option<ServerSettingDAO>> {
        let setting: Option<ServerSettingDAO> = sqlx::query_as(
            "SELECT security_disable_client_side_telemetry, network_external_url, security_allowed_register_domain_list, billing_enterprise_license
            FROM server_setting WHERE id = ?;"
        ).bind(SERVER_SETTING_ROW_ID)
        .fetch_optional(&mut **transaction)
        .await?;
        Ok(setting)
    }

    pub async fn read_server_setting(&self) -> Result<ServerSettingDAO> {
        let mut transaction = self.pool.begin().await?;
        let setting = self.internal_read_server_setting(&mut transaction).await?;
        let Some(setting) = setting else {
            query!(
                "INSERT INTO server_setting (id) VALUES (?);",
                SERVER_SETTING_ROW_ID
            )
            .execute(&mut *transaction)
            .await?;
            let setting = self
                .internal_read_server_setting(&mut transaction)
                .await?
                .expect("Freshly-written row must always be present");
            transaction.commit().await?;
            return Ok(setting);
        };
        Ok(setting)
    }

    pub async fn update_security_setting(
        &self,
        allowed_register_domain_list: Option<String>,
        disable_client_side_telemetry: bool,
    ) -> Result<()> {
        query!("INSERT INTO server_setting (id, security_allowed_register_domain_list, security_disable_client_side_telemetry) VALUES ($1, $2, $3)
                ON CONFLICT(id) DO UPDATE SET security_allowed_register_domain_list = $2, security_disable_client_side_telemetry = $3",
            SERVER_SETTING_ROW_ID,
            allowed_register_domain_list,
            disable_client_side_telemetry,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn update_network_setting(&self, external_url: String) -> Result<()> {
        query!(
            "INSERT INTO server_setting (id, network_external_url) VALUES ($1, $2)
                ON CONFLICT(id) DO UPDATE SET network_external_url = $2",
            SERVER_SETTING_ROW_ID,
            external_url
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn read_enterprise_license(&self) -> Result<Option<String>> {
        Ok(sqlx::query_scalar(
            "SELECT billing_enterprise_license FROM server_setting WHERE id = ?;",
        )
        .bind(SERVER_SETTING_ROW_ID)
        .fetch_one(&self.pool)
        .await?)
    }

    pub async fn update_enterprise_license(
        &self,
        enterprise_license: Option<String>,
    ) -> Result<()> {
        query!(
            "INSERT INTO server_setting (id, billing_enterprise_license) VALUES ($1, $2)
                    ON CONFLICT(id) DO UPDATE SET billing_enterprise_license = $2",
            SERVER_SETTING_ROW_ID,
            enterprise_license
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dao(security_allowed_register_domain_list: Option<String>) -> ServerSettingDAO {
        ServerSettingDAO {
            billing_enterprise_license: None,
            security_allowed_register_domain_list,
            security_disable_client_side_telemetry: false,
            network_external_url: "http://localhost:8080".into(),
        }
    }
    #[test]
    fn test_security_allowed_register_domain_list() {
        let dao = make_dao(None);
        let domains: Vec<_> = dao.security_allowed_register_domain_list().collect();
        assert!(domains.is_empty());

        let dao = make_dao(Some("".into()));
        let domains: Vec<_> = dao.security_allowed_register_domain_list().collect();
        assert!(domains.is_empty());

        let dao = make_dao(Some("    ".into()));
        let domains: Vec<_> = dao.security_allowed_register_domain_list().collect();
        assert!(domains.is_empty());

        let dao = make_dao(Some("abc.com, def.com".into()));
        let domains: Vec<_> = dao.security_allowed_register_domain_list().collect();
        assert_eq!(domains[0], "abc.com");
        assert_eq!(domains[1], "def.com");
    }
}
