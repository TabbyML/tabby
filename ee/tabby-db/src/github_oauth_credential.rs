use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension};

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
        client_secret: &str,
        active: bool,
    ) -> Result<()> {
        let client_id = client_id.to_string();
        let client_secret = client_secret.to_string();

        self.conn
            .call(move |c| {
                let mut stmt = c.prepare(
                    r#"INSERT INTO github_oauth_credential (id, client_id, client_secret)
                    VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE
                    SET client_id = ?, client_secret = ?, active = ?, updated_at = datetime('now')
                    WHERE id = ?"#,
                )?;
                stmt.insert(params![
                    GITHUB_OAUTH_CREDENTIAL_ROW_ID, client_id, client_secret,
                    client_id, client_secret, active,
                    GITHUB_OAUTH_CREDENTIAL_ROW_ID
                ])?;
                Ok(())
            })
            .await?;

        Ok(())
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
        // test insert
        let conn = DbConn::new().await.unwrap();
        conn.update_github_oauth_credential("client_id", "client_secret", false)
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");
        assert_eq!(res.active, true);

        // test update
        let id = conn
            .update_github_oauth_credential("client_id", "client_secret_2", false)
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret_2");
        assert_eq!(res.active, false);
    }
}
