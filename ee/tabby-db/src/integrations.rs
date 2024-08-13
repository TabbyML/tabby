use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(FromRow)]
pub struct IntegrationDAO {
    pub id: i64,
    pub kind: String,
    pub error: Option<String>,
    pub display_name: String,
    pub access_token: String,
    pub api_base: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub synced: bool,
}

impl DbConn {
    pub async fn create_integration(
        &self,
        kind: String,
        name: String,
        access_token: String,
        api_base: Option<String>,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO integrations(kind, display_name, access_token, api_base) VALUES (?, ?, ?, ?);",
            kind,
            name,
            access_token,
            api_base
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn get_integration(&self, id: i64) -> Result<IntegrationDAO> {
        let provider = query_as!(
            IntegrationDAO,
            r#"SELECT
                id,
                kind,
                error,
                display_name,
                access_token,
                api_base,
                updated_at as "updated_at!: DateTime<Utc>",
                created_at as "created_at!: DateTime<Utc>",
                synced
            FROM integrations WHERE id = ?;"#,
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(provider)
    }

    pub async fn delete_integration(&self, id: i64, kind: &str) -> Result<()> {
        let res = query!(
            "DELETE FROM integrations WHERE id = ? AND kind = ?;",
            id,
            kind
        )
        .execute(&self.pool)
        .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No integration access token to delete"));
        }
        Ok(())
    }

    pub async fn update_integration(
        &self,
        id: i64,
        kind: &str,
        display_name: String,
        access_token: Option<String>,
        api_base: Option<String>,
    ) -> Result<()> {
        let access_token = match access_token {
            Some(access_token) => access_token,
            None => self.get_integration(id).await?.access_token,
        };

        let res = query!(
            "UPDATE integrations SET display_name = $1, access_token = $2, api_base = $3, updated_at = DATETIME('now'),
                synced = IIF(access_token != $2 OR api_base != $3, false, synced),
                error = IIF(access_token != $2 OR api_base != $3, NULL, error)
            WHERE id = $4 AND kind = $5;",
            display_name,
            access_token,
            api_base,
            id,
            kind
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!(
                "The specified integration access token does not exist"
            ));
        }

        Ok(())
    }

    pub async fn update_integration_error(&self, id: i64, error: Option<String>) -> Result<()> {
        query!(
            "UPDATE integrations SET synced = true, error = ? WHERE id = ?",
            error,
            id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_integrations(
        &self,
        ids: Vec<i64>,
        kind: Option<String>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<IntegrationDAO>> {
        let mut conditions = vec![];

        let id_condition = (!ids.is_empty()).then(|| {
            let ids = ids
                .into_iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("id in ({ids})")
        });
        conditions.extend(id_condition);

        let kind_condition = kind.map(|kind| format!("kind = '{kind}'"));
        conditions.extend(kind_condition);

        let condition = (!conditions.is_empty()).then(|| conditions.join(" AND "));

        let providers = query_paged_as!(
            IntegrationDAO,
            "integrations",
            [
                "id",
                "kind",
                "error",
                "display_name",
                "access_token",
                "api_base",
                "created_at" as "created_at!: DateTime<Utc>",
                "updated_at" as "updated_at!: DateTime<Utc>",
                "synced"
            ],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(providers)
    }
}
