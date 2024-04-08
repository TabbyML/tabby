use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;
use tabby_db::DbConn;
use tracing::warn;

use super::AsRowid;
use crate::schema::{
    analytic::{AnalyticService, CompletionStats, Language},
    Result,
};

struct AnalyticServiceImpl {
    db: DbConn,
}

#[async_trait]
impl AnalyticService for AnalyticServiceImpl {
    async fn daily_stats_in_past_year(&self, users: Vec<ID>) -> Result<Vec<CompletionStats>> {
        let users = convert_ids(users);
        let stats = self.db.compute_daily_stats_in_past_year(users).await?;
        let stats = stats
            .into_iter()
            .map(|s| CompletionStats {
                start: s.start,
                end: s.start + chrono::Duration::days(1),
                completions: s.completions,
                selects: s.selects,
            })
            .collect();
        Ok(stats)
    }

    async fn daily_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<ID>,
        mut languages: Vec<Language>,
    ) -> Result<Vec<CompletionStats>> {
        let users = convert_ids(users);

        let include_other_languages = languages.iter().any(|l| l == &Language::Other);
        let not_languages = if include_other_languages {
            Some(Language::all_known().flat_map(|l| l.to_strings()).collect())
        } else {
            None
        };

        languages.retain(|l| l != &Language::Other);

        let languages = languages.into_iter().flat_map(|l| l.to_strings()).collect();
        let stats = self
            .db
            .compute_daily_stats(
                start,
                end,
                users,
                languages,
                not_languages.unwrap_or_default(),
            )
            .await?;
        let stats = stats
            .into_iter()
            .map(|s| CompletionStats {
                start: s.start,
                end: s.start + chrono::Duration::days(1),
                completions: s.completions,
                selects: s.selects,
            })
            .collect();
        Ok(stats)
    }
}

fn convert_ids(ids: Vec<ID>) -> Vec<i64> {
    ids.into_iter()
        .filter_map(|id| match id.as_rowid() {
            Ok(rowid) => Some(rowid),
            Err(_) => {
                warn!("Ignoring invalid ID: {}", id);
                None
            }
        })
        .collect()
}

pub fn new_analytic_service(db: DbConn) -> Arc<dyn AnalyticService> {
    Arc::new(AnalyticServiceImpl { db })
}

#[cfg(test)]
mod tests {
    use chrono::Days;

    use super::*;
    use crate::service::AsID;

    fn timestamp() -> u128 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let start = SystemTime::now();
        start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis()
    }

    #[tokio::test]
    async fn test_daily_stats_in_past_year() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = db
            .create_user("test@example.com".into(), Some("pass".into()), true)
            .await
            .unwrap();

        let completion_id = "completion_id".to_string();
        db.create_user_completion(
            timestamp(),
            user_id,
            completion_id.clone(),
            "lang".to_string(),
        )
        .await
        .unwrap();

        db.add_to_user_completion(timestamp(), &completion_id, 0, 1, 0)
            .await
            .unwrap();

        let svc = new_analytic_service(db);
        let activity = svc
            .daily_stats_in_past_year(vec![user_id.as_id()])
            .await
            .unwrap();
        assert_eq!(1, activity.len());
        assert_eq!(1, activity[0].completions);
        assert_eq!(1, activity[0].selects);
    }

    #[tokio::test]
    async fn test_daily_stats() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = db
            .create_user("test@example.com".into(), Some("pass".into()), true)
            .await
            .unwrap();

        let completion_id = "completion_id".to_string();
        db.create_user_completion(
            timestamp(),
            user_id,
            completion_id.clone(),
            "rust".to_string(),
        )
        .await
        .unwrap();

        db.add_to_user_completion(timestamp(), &completion_id, 0, 1, 0)
            .await
            .unwrap();

        let svc = new_analytic_service(db);
        let end = Utc::now();
        let start = end.checked_sub_days(Days::new(100)).unwrap();
        let stats = svc.daily_stats(start, end, vec![], vec![]).await.unwrap();
        assert_eq!(1, stats.len());
        assert_eq!(1, stats[0].completions);
        assert_eq!(1, stats[0].selects);
    }

    #[tokio::test]
    async fn test_other_langs() {
        let db = DbConn::new_in_memory().await.unwrap();

        let user_id = db
            .create_user("test@example.com".into(), Some("pass".into()), true)
            .await
            .unwrap();

        db.create_user_completion(
            timestamp(),
            user_id,
            "completion_id".into(),
            "testlang".into(),
        )
        .await
        .unwrap();

        db.create_user_completion(timestamp(), user_id, "completion_id2".into(), "rust".into())
            .await
            .unwrap();

        let service = new_analytic_service(db);
        let end = Utc::now();
        let start = end.checked_sub_days(Days::new(100)).unwrap();

        let stats = service
            .daily_stats(start, end, vec![], vec![Language::Other])
            .await
            .unwrap();

        assert_eq!(1, stats.len());
        assert_eq!(1, stats[0].completions);
    }
}
