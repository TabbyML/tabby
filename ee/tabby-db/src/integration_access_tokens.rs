use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::{DateTimeUtc, DbConn};

#[derive(FromRow)]
pub struct IntegrationAccessTokenDAO {
    pub id: i64,
    pub kind: String,
    pub error: Option<String>,
    pub display_name: String,
    pub access_token: String,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

impl DbConn {
    pub async fn create_integration_access_token(
        &self,
        kind: String,
        name: String,
        access_token: String,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO integration_access_tokens(kind, display_name, access_token) VALUES (?, ?, ?);",
            kind,
            name,
            access_token
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn get_integration_access_token(&self, id: i64) -> Result<IntegrationAccessTokenDAO> {
        let provider = query_as!(
            IntegrationAccessTokenDAO,
            r#"SELECT
                id,
                kind,
                error,
                display_name,
                access_token,
                created_at AS "created_at: DateTimeUtc",
                updated_at AS "updated_at: DateTimeUtc"
            FROM integration_access_tokens WHERE id = ?;"#,
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(provider)
    }

    pub async fn delete_integration_access_token(&self, id: i64) -> Result<()> {
        let res = query!("DELETE FROM integration_access_tokens WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No integration access token to delete"));
        }
        Ok(())
    }

    pub async fn update_integration_access_token(
        &self,
        id: i64,
        display_name: String,
        access_token: Option<String>,
    ) -> Result<()> {
        let access_token = match access_token {
            Some(access_token) => access_token,
            None => self.get_integration_access_token(id).await?.access_token,
        };

        let res = query!(
            "UPDATE integration_access_tokens SET display_name = ?, access_token=? WHERE id = ?;",
            display_name,
            access_token,
            id
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

    pub async fn update_integration_access_token_error(
        &self,
        id: i64,
        error: Option<String>,
    ) -> Result<()> {
        query!(
            "UPDATE integration_access_tokens SET updated_at = DATETIME('now'), error = ? WHERE id = ?",
            error,
            id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_integration_access_tokens(
        &self,
        ids: Vec<i64>,
        kind: Option<String>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<IntegrationAccessTokenDAO>> {
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
            IntegrationAccessTokenDAO,
            "integration_access_tokens",
            [
                "id",
                "kind",
                "error",
                "display_name",
                "access_token",
                "created_at" as "created_at: DateTimeUtc",
                "updated_at" as "updated_at: DateTimeUtc"
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
