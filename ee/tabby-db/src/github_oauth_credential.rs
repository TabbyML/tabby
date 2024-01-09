use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{named_params, OptionalExtension};

use super::DbConn;

const GITHUB_OAUTH_CREDENTIAL_ROW_ID: i32 = 1;

pub struct GithubOAuthCredentialDAO {
    pub client_id: String,
    pub client_secret: String,
    pub active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl GithubOAuthCredentialDAO {
    fn from_row(row: &rusqlite::Row<'_>) -> std::result::Result<Self, rusqlite::Error> {
        Ok(Self {
            client_id: row.get(0)?,
            client_secret: row.get(1)?,
            active: row.get(2)?,
            created_at: row.get(3)?,
            updated_at: row.get(4)?,
        })
    }
}

/// db read/write operations for `github_oauth_credential` table
impl DbConn {
    pub async fn update_github_oauth_credential(
        &self,
        client_id: &str,
        client_secret: Option<&str>,
        active: bool,
    ) -> Result<()> {
        let client_id = client_id.to_string();
        if let Some(client_secret) = client_secret {
            let client_secret = client_secret.to_string();
            let sql = r#"INSERT INTO github_oauth_credential (id, client_id, client_secret, active)
                                VALUES (:id, :cid, :secret, :active) ON CONFLICT(id) DO UPDATE
                                SET client_id = :cid, client_secret = :secret, active = :active, updated_at = datetime('now')
                                WHERE id = :id"#;
            self.conn
                .call(move |c| {
                    let mut stmt = c.prepare(sql)?;
                    stmt.insert(named_params! {
                    ":id": GITHUB_OAUTH_CREDENTIAL_ROW_ID,
                    ":cid": client_id,
                    ":secret": client_secret,
                    ":active": active,
                    })?;
                    Ok(())
                })
                .await?;
            Ok(())
        } else {
            let sql = r#"
            UPDATE github_oauth_credential SET client_id = :cid, active = :active, updated_at = datetime('now')
            WHERE id = :id"#;
            let rows = self
                .conn
                .call(move |c| {
                    let mut stmt = c.prepare(sql)?;
                    let rows = stmt.execute(named_params! {
                    ":id": GITHUB_OAUTH_CREDENTIAL_ROW_ID,
                    ":cid": client_id,
                    ":active": active,
                    })?;
                    Ok(rows)
                })
                .await?;
            if rows != 1 {
                return Err(anyhow::anyhow!(
                    "failed to update: github credential not found"
                ));
            }
            Ok(())
        }
    }

    pub async fn read_github_oauth_credential(&self) -> Result<Option<GithubOAuthCredentialDAO>> {
        let token = self
            .conn
            .call(|conn| {
                Ok(conn
                    .query_row(
                        r#"SELECT client_id, client_secret, active, created_at, updated_at FROM github_oauth_credential WHERE id = ?"#,
                        [GITHUB_OAUTH_CREDENTIAL_ROW_ID],
                        GithubOAuthCredentialDAO::from_row,
                    )
                    .optional())
            })
            .await?;

        Ok(token?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_update_github_oauth_credential() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // test update failure when no record exists
        let res = conn
            .update_github_oauth_credential("client_id", None, false)
            .await;
        assert!(res.is_err());

        // test insert
        conn.update_github_oauth_credential("client_id", Some("client_secret"), true)
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");
        assert!(res.active);

        // test update
        conn.update_github_oauth_credential("client_id", Some("client_secret_2"), false)
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret_2");
        assert!(!res.active);

        // test update without client_secret
        conn.update_github_oauth_credential("client_id_2", None, true)
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id_2");
        assert_eq!(res.client_secret, "client_secret_2");
        assert!(res.active);
    }
}
