use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{named_params, OptionalExtension};

use super::DbConn;

const GOOGLE_OAUTH_CREDENTIAL_ROW_ID: i32 = 1;

pub struct GoogleOAuthCredentialDAO {
    pub client_id: String,
    pub client_secret: String,
    pub redirect_uri: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl GoogleOAuthCredentialDAO {
    fn from_row(row: &rusqlite::Row<'_>) -> std::result::Result<Self, rusqlite::Error> {
        Ok(Self {
            client_id: row.get(0)?,
            client_secret: row.get(1)?,
            redirect_uri: row.get(2)?,
            created_at: row.get(3)?,
            updated_at: row.get(4)?,
        })
    }
}

/// db read/write operations for `google_oauth_credential` table
impl DbConn {
    pub async fn update_google_oauth_credential(
        &self,
        client_id: &str,
        client_secret: &str,
        redirect_uri: Option<&str>,
    ) -> Result<()> {
        let redirect_uri = redirect_uri.unwrap_or_default().to_string();
        if !client_secret.is_empty() {
            let client_id = client_id.to_string();
            let client_secret = client_secret.to_string();
            let sql = r#"INSERT INTO google_oauth_credential (id, client_id, client_secret, redirect_uri)
                                VALUES (:id, :cid, :secret, :redirect) ON CONFLICT(id) DO UPDATE
                                SET client_id = :cid, client_secret = :secret, redirect_uri = :redirect, updated_at = datetime('now')
                                WHERE id = :id"#;
            self.conn
                .call(move |c| {
                    let mut stmt = c.prepare(sql)?;
                    stmt.insert(named_params! {
                    ":id": GOOGLE_OAUTH_CREDENTIAL_ROW_ID,
                    ":cid": client_id,
                    ":secret": client_secret,
                    ":redirect": redirect_uri,
                    })?;
                    Ok(())
                })
                .await?;
            Ok(())
        } else {
            let sql = r#"
            UPDATE google_oauth_credential SET redirect_uri = :redirect, updated_at = datetime('now')
            WHERE id = :id"#;
            let rows = self
                .conn
                .call(move |c| {
                    let mut stmt = c.prepare(sql)?;
                    let rows = stmt.execute(named_params! {
                    ":id": GOOGLE_OAUTH_CREDENTIAL_ROW_ID,
                    ":redirect": redirect_uri,
                    })?;
                    Ok(rows)
                })
                .await?;
            if rows != 1 {
                return Err(anyhow::anyhow!(
                    "failed to update: google credential not found"
                ));
            }
            Ok(())
        }
    }

    pub async fn read_google_oauth_credential(&self) -> Result<Option<GoogleOAuthCredentialDAO>> {
        let token = self
            .conn
            .call(|conn| {
                Ok(conn
                    .query_row(
                        r#"SELECT client_id, client_secret, redirect_uri, created_at, updated_at FROM google_oauth_credential WHERE id = ?"#,
                        [GOOGLE_OAUTH_CREDENTIAL_ROW_ID],
                        GoogleOAuthCredentialDAO::from_row,
                    )
                    .optional())
            })
            .await?;

        Ok(token?)
    }

    pub async fn delete_google_oauth_credential(&self) -> Result<()> {
        Ok(self
            .conn
            .call(move |c| {
                c.execute(
                    "DELETE FROM google_oauth_credential WHERE id = ?",
                    [GOOGLE_OAUTH_CREDENTIAL_ROW_ID],
                )?;
                Ok(())
            })
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_update_google_oauth_credential() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // test update failure when no record exists
        let res = conn
            .update_google_oauth_credential("", "", Some("http://localhost"))
            .await;
        assert!(res.is_err());

        // test insert
        conn.update_google_oauth_credential("client_id", "client_secret", None)
            .await
            .unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");
        assert_eq!(res.redirect_uri, "");

        // test delete
        conn.delete_google_oauth_credential().await.unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap();
        assert!(res.is_none());

        // test insert with redirect_uri
        conn.update_google_oauth_credential("client_id", "client_secret", Some("http://localhost"))
            .await
            .unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");
        assert_eq!(res.redirect_uri, "http://localhost");

        // test update
        conn.update_google_oauth_credential(
            "client_id_2",
            "client_secret_2",
            Some("http://127.0.0.1"),
        )
        .await
        .unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id_2");
        assert_eq!(res.client_secret, "client_secret_2");
        assert_eq!(res.redirect_uri, "http://127.0.0.1");

        // test update redirect_uri
        conn.update_google_oauth_credential("", "", Some("http://localhost"))
            .await
            .unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id_2");
        assert_eq!(res.client_secret, "client_secret_2");
        assert_eq!(res.redirect_uri, "http://localhost");
    }
}
