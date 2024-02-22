use std::fmt::Display;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query};
use tabby_common::api::event::EventLogger;

use crate::DbConn;

// TODO: add user completions dao.
#[derive(FromRow)]
pub struct UserCompletionDAO {
    pub user_id: i32,
    pub completion_id: String,
    pub language: String,

    pub views: i64,
    pub selects: i64,
    pub dismisses: i64,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn create_user_completion(
        &self,
        user_id: i32,
        completion_id: String,
        language: String,
    ) -> Result<i32> {
        let res = query!(
            "INSERT INTO user_completions (user_id, completion_id, language) VALUES (?, ?, ?);",
            user_id,
            completion_id,
            language
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid() as i32)
    }

    pub async fn add_to_user_completion(
        &self,
        completion_id: &str,
        views: i64,
        selects: i64,
        dismisses: i64,
    ) -> Result<()> {
        query!("UPDATE user_completions SET views = views + ?, selects = selects + ?, dismisses = dismisses + ? WHERE completion_id = ?",
            views, selects, dismisses, completion_id).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn list_user_completions_with_filter(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<UserCompletionDAO>> {
        let query = Self::make_pagination_query(
            "user_completions",
            &[
                "id",
                "user_id",
                "completion_id",
                "language",
                "views",
                "selects",
                "dismisses",
                "created_at",
                "updated_at",
            ],
            limit,
            skip_id,
            backwards,
        );
        let completions = sqlx::query_as(&query).fetch_all(&self.pool).await?;
        Ok(completions)
    }
}

fn log_err<T, E: Display>(res: Result<T, E>) {
    if let Err(e) = res {
        eprintln!("Failed to log event: {e}");
    }
}

#[async_trait]
impl EventLogger for DbConn {
    async fn log(&self, e: tabby_common::api::event::Event) {
        match e {
            tabby_common::api::event::Event::View { completion_id, .. } => {
                log_err(self.add_to_user_completion(&completion_id, 1, 0, 0).await)
            }
            tabby_common::api::event::Event::Select { completion_id, .. } => {
                log_err(self.add_to_user_completion(&completion_id, 0, 1, 0).await)
            }
            tabby_common::api::event::Event::Dismiss { completion_id, .. } => {
                log_err(self.add_to_user_completion(&completion_id, 0, 0, 1).await)
            }
            tabby_common::api::event::Event::Completion {
                completion_id,
                language,
                user,
                ..
            } => {
                let Some(user) = user else { return };
                let user_db = self.get_user_by_email(&user).await;
                let Ok(Some(user_db)) = user_db else {
                    eprintln!("Failed to retrieve user for {user}");
                    return;
                };
                log_err(
                    self.create_user_completion(user_db.id, completion_id, language)
                        .await,
                );
            }
            tabby_common::api::event::Event::ChatCompletion { .. } => {}
        }
    }
}
