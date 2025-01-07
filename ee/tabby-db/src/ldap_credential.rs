use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::query;

use crate::DbConn;

pub struct LdapCredentialDAO {
    pub host: String,
    pub port: i64,

    pub bind_dn: String,
    pub bind_password: String,
    pub base_dn: String,
    pub user_filter: String,

    pub encryption: String,
    pub skip_tls_verify: bool,

    pub email_attribute: String,
    pub name_attribute: Option<String>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn update_ldap_credential(
        &self,
        host: &str,
        port: i32,
        bind_dn: &str,
        bind_password: &str,
        base_dn: &str,
        user_filter: &str,
        encryption: &str,
        skip_tls_verify: bool,
        email_attribute: &str,
        name_attribute: Option<&str>,
    ) -> Result<()> {
        // only support one ldap credential, so id is always 1
        query!(
            r#"INSERT INTO ldap_credential (
                id,
                host,
                port,
                bind_dn,
                bind_password,
                base_dn,
                user_filter,
                encryption,
                skip_tls_verify,
                email_attribute,
                name_attribute
            )
            VALUES (1, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                host = excluded.host,
                port = excluded.port,
                bind_dn = excluded.bind_dn,
                bind_password = excluded.bind_password,
                base_dn = excluded.base_dn,
                user_filter = excluded.user_filter,
                encryption = excluded.encryption,
                skip_tls_verify = excluded.skip_tls_verify,
                email_attribute = excluded.email_attribute,
                name_attribute = excluded.name_attribute,
                updated_at = datetime('now')"#,
            host,
            port,
            bind_dn,
            bind_password,
            base_dn,
            user_filter,
            encryption,
            skip_tls_verify,
            email_attribute,
            name_attribute,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete_ldap_credential(&self) -> Result<()> {
        // only support one ldap credential, so id is always 1
        query!("DELETE FROM ldap_credential WHERE id = 1")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn read_ldap_credential(&self) -> Result<Option<LdapCredentialDAO>> {
        let token = sqlx::query_as!(
            LdapCredentialDAO,
            r#"SELECT
                  host as "host: String",
                  port,
                  bind_dn as "bind_dn: String",
                  bind_password as "bind_password: String",
                  base_dn as "base_dn: String",
                  user_filter as "user_filter: String",
                  encryption as "encryption: String",
                  skip_tls_verify,
                  email_attribute as "email_attribute: String",
                  name_attribute as "name_attribute: String",
                  created_at as "created_at: DateTime<Utc>",
                  updated_at as "updated_at: DateTime<Utc>"
               FROM ldap_credential
               WHERE id = 1"#,
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_update_ldap_credential() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // test insert
        conn.update_ldap_credential(
            "host",
            389,
            "bind_dn",
            "bind_password",
            "base_dn",
            "user_filter",
            "encryption",
            true,
            "email_attribute",
            Some("name_attribute"),
        )
        .await
        .unwrap();
        let res = conn.read_ldap_credential().await.unwrap().unwrap();
        assert_eq!(res.host, "host");
        assert_eq!(res.port, 389);
        assert_eq!(res.bind_dn, "bind_dn");
        assert_eq!(res.bind_password, "bind_password");
        assert_eq!(res.base_dn, "base_dn");
        assert_eq!(res.user_filter, "user_filter");
        assert_eq!(res.encryption, "encryption");
        assert!(res.skip_tls_verify);
        assert_eq!(res.email_attribute, "email_attribute");
        assert_eq!(res.name_attribute, Some("name_attribute".into()));
        let created_at = res.created_at;
        let updated_at = res.updated_at;
        assert!(created_at > Utc::now() - chrono::Duration::seconds(2));
        assert!(updated_at > Utc::now() - chrono::Duration::seconds(2));

        // test update
        // sleep for a while to make sure updated_at is different
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        conn.update_ldap_credential(
            "host_2",
            389,
            "bind_dn_2",
            "bind_password_2",
            "base_dn_2",
            "user_filter_2",
            "encryption_2",
            false,
            "email_attribute_2",
            Some("name_attribute_2"),
        )
        .await
        .unwrap();
        let res = conn.read_ldap_credential().await.unwrap().unwrap();
        assert_eq!(res.host, "host_2");
        assert_eq!(res.port, 389);
        assert_eq!(res.bind_dn, "bind_dn_2");
        assert_eq!(res.bind_password, "bind_password_2");
        assert_eq!(res.base_dn, "base_dn_2");
        assert_eq!(res.user_filter, "user_filter_2");
        assert_eq!(res.encryption, "encryption_2");
        assert!(!res.skip_tls_verify);
        assert_eq!(res.email_attribute, "email_attribute_2");
        assert_eq!(res.name_attribute, Some("name_attribute_2".into()));
        assert_eq!(res.created_at, created_at);
        assert!(res.updated_at > updated_at);

        // make sure only one row in the table
        let res = sqlx::query_scalar!("SELECT COUNT(*) FROM ldap_credential")
            .fetch_one(&conn.pool)
            .await
            .unwrap();
        assert_eq!(res, 1);
    }

    #[tokio::test]
    async fn test_delete_ldap_credential() {
        let conn = DbConn::new_in_memory().await.unwrap();

        conn.update_ldap_credential(
            "host",
            389,
            "bind_dn",
            "bind_password",
            "base_dn",
            "user_filter",
            "encryption",
            true,
            "email_attribute",
            Some("name_attribute"),
        )
        .await
        .unwrap();

        // make sure inserted
        let res = conn.read_ldap_credential().await.unwrap();
        assert!(res.is_some());

        // delete
        conn.delete_ldap_credential().await.unwrap();
        let res = conn.read_ldap_credential().await.unwrap();
        assert!(res.is_none());
    }
}
