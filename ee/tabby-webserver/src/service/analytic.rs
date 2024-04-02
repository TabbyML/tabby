use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;
use tabby_db::DbConn;

use crate::schema::analytic::{AnalyticService, CompletionStats, Language};
use crate::schema::Result;

struct AnalyticServiceImpl {
    db: DbConn,
}

#[async_trait]
impl AnalyticService for AnalyticServiceImpl {
    // FIXME(boxbeam): Implementing in memory caching with 1 hour expiry.
    async fn annual_activity(&self) -> Result<Vec<CompletionStats>> {
        let stats = self.db.compute_annual_activity().await?;
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

    async fn daily_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<ID>,
        languages: Vec<Language>,
    ) -> Result<CompletionStats> {
        todo!()
    }
}

pub fn new_analytic_service(db: DbConn) -> Arc<dyn AnalyticService> {
    Arc::new(AnalyticServiceImpl { db })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timestamp() -> u128 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let start = SystemTime::now();
        start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis()
    }

    #[tokio::test]
    async fn test_annual_activity() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = db
            .create_user("test@example.com".into(), Some("pass".into()), true)
            .await
            .unwrap();

        let completion_id = "completion_id".to_string();
        db.create_user_completion(timestamp(), user_id, completion_id.clone(), "lang".to_string())
            .await
            .unwrap();

        db.add_to_user_completion(timestamp(), &completion_id, 0, 1, 0).await.unwrap();

        let svc = new_analytic_service(db);
        let activity = svc.annual_activity().await.unwrap();
        assert_eq!(1, activity.len());
        assert_eq!(1, activity[0].completions);
        assert_eq!(1, activity[0].selects);
    }
}
