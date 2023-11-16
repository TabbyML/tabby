mod proxy;
mod worker;

use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use axum::{http::Request, middleware::Next, response::IntoResponse};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use rusqlite::params;
use tokio_rusqlite::Connection;
use tracing::{info, warn};

use crate::api::{HubError, Worker, WorkerKind};

pub struct ServerContext {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
    db_conn: Arc<Connection>,
}

impl ServerContext {
    pub fn new(db_conn: Arc<Connection>) -> Self {
        Self {
            client: Client::default(),
            completion: worker::WorkerGroup::default(),
            chat: worker::WorkerGroup::default(),
            db_conn,
        }
    }

    /// Query token from database.
    /// Since token is global unique for each tabby server, by right there's only one row in the table.
    pub async fn token(&self) -> Result<String> {
        let token = self
            .db_conn
            .call(|conn| {
                
                conn.query_row(r#"SELECT token FROM token_tab WHERE id = 1"#, [], |row| {
                        row.get(0)
                    })
            })
            .await?;

        Ok(token)
    }

    /// Replace old token with new token.
    /// The old token is used to verify the request,
    /// only when old token from the request matches the one in the database,
    /// then it's replaced with the new one.
    pub async fn refresh_token(&self, old: String, new: String) -> Result<()> {
        if new.is_empty() {
            return Err(anyhow::anyhow!("failed: new token is empty"));
        }
        let updated_at = chrono::Utc::now().timestamp() as u32;
        let res = self
            .db_conn
            .call(move |conn| {
                
                conn.execute(
                    r#"UPDATE token_tab SET token = ?, updated_at = ? WHERE token = ?"#,
                    params![new, updated_at, old],
                )
            })
            .await?;
        if res != 1 {
            return Err(anyhow::anyhow!("failed: mismatched old token"));
        }
        Ok(())
    }

    pub async fn register_worker(&self, worker: Worker) -> Result<Worker, HubError> {
        let worker = match worker.kind {
            WorkerKind::Completion => self.completion.register(worker).await,
            WorkerKind::Chat => self.chat.register(worker).await,
        };

        if let Some(worker) = worker {
            info!(
                "registering <{:?}> worker running at {}",
                worker.kind, worker.addr
            );
            Ok(worker)
        } else {
            Err(HubError::RequiresEnterpriseLicense)
        }
    }

    pub async fn list_workers(&self) -> Vec<Worker> {
        [self.completion.list().await, self.chat.list().await].concat()
    }

    pub async fn dispatch_request(
        &self,
        request: Request<Body>,
        next: Next<Body>,
    ) -> axum::response::Response {
        let path = request.uri().path();

        let remote_addr = request
            .extensions()
            .get::<axum::extract::ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0)
            .expect("Unable to extract remote addr");

        let worker = if path.starts_with("/v1/completions") {
            self.completion.select().await
        } else if path.starts_with("/v1beta/chat/completions") {
            self.chat.select().await
        } else {
            None
        };

        if let Some(worker) = worker {
            match proxy::call(self.client.clone(), remote_addr.ip(), &worker, request).await {
                Ok(res) => res.into_response(),
                Err(err) => {
                    warn!("Failed to proxy request {}", err);
                    axum::response::Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .unwrap()
                        .into_response()
                }
            }
        } else {
            next.run(request).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    #[tokio::test]
    async fn test_token() {
        let conn = db::init_memory_db().await;
        let ctx = ServerContext::new(Arc::new(conn));
        let token = ctx.token().await.unwrap();
        assert_eq!(token, "");
    }

    #[tokio::test]
    async fn test_refresh_token() {
        let conn = db::init_memory_db().await;
        let ctx = ServerContext::new(Arc::new(conn));

        // first refresh
        let new1 = "new_token_1".to_string();
        ctx.refresh_token("".to_string(), new1.clone())
            .await
            .unwrap();
        let token = ctx.token().await.unwrap();
        assert_eq!(token, new1);

        // second refresh
        let new2 = "new_token_2".to_string();
        ctx.refresh_token(new1.to_string(), new2.clone())
            .await
            .unwrap();
        let token = ctx.token().await.unwrap();
        assert_eq!(token, new2);

        // error case
        let res = ctx
            .refresh_token("invalid_token".to_string(), "new_token".to_string())
            .await;
        assert!(res.is_err());
    }
}
