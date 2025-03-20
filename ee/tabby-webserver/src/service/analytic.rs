use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    analytic::{
        AnalyticService, ChatCompletionStats, CompletionStats, DiskUsage, DiskUsageStats, Language,
    },
    AsID, Result,
};
use tracing::warn;

use super::AsRowid;

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
                end: (s.start + chrono::Duration::days(1)),
                language: s.language.into(),
                completions: s.completions,
                selects: s.selects,
                views: s.views,
            })
            .collect();
        Ok(stats)
    }

    async fn daily_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<ID>,
        languages: Vec<Language>,
    ) -> Result<Vec<CompletionStats>> {
        let users = convert_ids(users);

        let all_languages = Language::all_known()
            .flat_map(|l| l.language_names())
            .map(str::to_owned)
            .collect();
        let languages = languages
            .into_iter()
            .flat_map(|l| l.language_names())
            .map(str::to_owned)
            .collect();
        let stats = self
            .db
            .compute_daily_stats(start, end, users, languages, all_languages)
            .await?;
        let stats = stats
            .into_iter()
            .map(|s| CompletionStats {
                start: s.start,
                end: (s.start + chrono::Duration::days(1)),
                language: s.language.into(),
                completions: s.completions,
                selects: s.selects,
                views: s.views,
            })
            .collect();
        Ok(stats)
    }

    async fn chat_daily_stats_in_past_year(
        &self,
        users: Vec<ID>,
    ) -> Result<Vec<ChatCompletionStats>> {
        let users = convert_ids(users);
        let now = Utc::now();
        let stats = self
            .db
            .compute_chat_daily_stats(now - chrono::Duration::days(365), now, users)
            .await?;
        let stats = stats
            .into_iter()
            .map(|s| ChatCompletionStats {
                start: s.start,
                end: (s.start + chrono::Duration::days(1)),
                user_id: s.user_id.as_id(),
                chats: s.chats,
            })
            .collect();
        Ok(stats)
    }

    async fn chat_daily_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<ID>,
    ) -> Result<Vec<ChatCompletionStats>> {
        let users = convert_ids(users);

        let stats = self.db.compute_chat_daily_stats(start, end, users).await?;
        let stats = stats
            .into_iter()
            .map(|s| ChatCompletionStats {
                start: s.start,
                end: (s.start + chrono::Duration::days(1)),
                user_id: s.user_id.as_id(),
                chats: s.chats,
            })
            .collect();
        Ok(stats)
    }

    async fn disk_usage_stats(&self) -> Result<DiskUsageStats> {
        Ok(DiskUsageStats {
            events: dir_size(tabby_common::path::events_dir()).await?,
            indexed_repositories: dir_size(tabby_common::path::index_dir()).await?,
            database: dir_size(crate::path::tabby_ee_root()).await?,
            models: dir_size(tabby_common::path::models_dir()).await?,
        })
    }
}

/// Calculate the size of a directory in kilobytes recursively
async fn dir_size(path: PathBuf) -> Result<DiskUsage, anyhow::Error> {
    let path_str = path.to_string_lossy().to_string();

    let size = if path.exists() {
        tokio::task::spawn_blocking(|| async { fs_extra::dir::get_size(path) })
            .await?
            .await?
    } else {
        0
    };

    Ok(DiskUsage {
        filepath: vec![path_str],
        size_kb: size as f64 / 1000.0,
    })
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
    use chrono::{Days, Duration};
    use tabby_common::path::set_tabby_root;
    use temp_testdir::TempDir;

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
            .create_user("test@example.com".into(), Some("pass".into()), true, None)
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

        let user_id2 = db
            .create_user("test2@example.com".into(), Some("pass".into()), false, None)
            .await
            .unwrap();

        db.create_user_completion(
            timestamp(),
            user_id2,
            "completion_id2".to_string(),
            "rust".to_string(),
        )
        .await
        .unwrap();

        // Query user 1 should return 1 completion and 1 select.
        let svc = new_analytic_service(db.clone());
        let activity = svc
            .daily_stats_in_past_year(vec![user_id.as_id()])
            .await
            .unwrap();

        assert_eq!(1, activity.len());
        assert_eq!(1, activity[0].completions);
        assert_eq!(1, activity[0].selects);

        // Query user 1 + user 2 should return 2 completions and 1 select.
        let activity = svc
            .daily_stats_in_past_year(vec![user_id.as_id(), user_id2.as_id()])
            .await
            .unwrap();

        assert_eq!(2, activity.len());
        assert_eq!(Language::Other, activity[0].language);
        assert_eq!(1, activity[0].completions);
        assert_eq!(1, activity[0].selects);
        assert_eq!(Language::Rust, activity[1].language);
        assert_eq!(1, activity[1].completions);
        assert_eq!(0, activity[1].selects);

        // Query all users should return 2 completions and 1 select.
        let activity = svc.daily_stats_in_past_year(vec![]).await.unwrap();

        assert_eq!(2, activity.len());
        assert_eq!(Language::Other, activity[0].language);
        assert_eq!(1, activity[0].completions);
        assert_eq!(1, activity[0].selects);
        assert_eq!(Language::Rust, activity[1].language);
        assert_eq!(1, activity[1].completions);
        assert_eq!(0, activity[1].selects);
    }

    #[tokio::test]
    async fn test_daily_stats() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = db
            .create_user("test@example.com".into(), Some("pass".into()), true, None)
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
        let end = Utc::now() + Duration::days(1);
        let start = end.checked_sub_days(Days::new(100)).unwrap();

        // Test that there is a single completion stat (1 day of history) with 1 completion and 1 select
        let stats = svc.daily_stats(start, end, vec![], vec![]).await.unwrap();
        assert_eq!(1, stats.len());
        assert_eq!(1, stats[0].completions);
        assert_eq!(1, stats[0].selects);

        // Test the same, but select only the specified user - regression test to prevent SQLite short-circuiting
        // from failing to account for `user_id` column not being present
        let stats = svc
            .daily_stats(start, end, vec![user_id.as_id()], vec![])
            .await
            .unwrap();
        assert_eq!(1, stats.len());
        assert_eq!(1, stats[0].completions);
        assert_eq!(1, stats[0].selects);
    }

    #[tokio::test]
    async fn test_other_langs() {
        let db = DbConn::new_in_memory().await.unwrap();

        let user_id = db
            .create_user("test@example.com".into(), Some("pass".into()), true, None)
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
        let end = Utc::now() + Duration::days(1);
        let start = end.checked_sub_days(Days::new(100)).unwrap();

        let stats = service
            .daily_stats(start, end, vec![], vec![Language::Rust, Language::Other])
            .await
            .unwrap();

        assert_eq!(2, stats.len());
        assert_eq!(Language::Other, stats[0].language);
        assert_eq!(1, stats[0].completions);
        assert_eq!(Language::Rust, stats[1].language);
        assert_eq!(1, stats[1].completions);

        let stats2 = service
            .daily_stats(start, end, vec![], vec![Language::Other])
            .await
            .unwrap();

        assert_eq!(1, stats2.len());
        assert_eq!(1, stats2[0].completions);
    }

    #[tokio::test]
    async fn test_inactive_user_gets_no_stats() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_analytic_service(db.clone());

        let id = db
            .create_user("testuser".into(), None, false, None)
            .await
            .unwrap();

        db.create_user_completion(timestamp(), id, "completion_id".into(), "rust".into())
            .await
            .unwrap();

        db.update_user_active(id, false).await.unwrap();

        assert!(service
            .daily_stats_in_past_year(vec![id.as_id()])
            .await
            .unwrap()
            .is_empty());

        let end = Utc::now() + Duration::days(1);
        let start = end.checked_sub_days(Days::new(100)).unwrap();

        assert!(service
            .daily_stats(start, end, vec![id.as_id()], vec![])
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn test_disk_usage() {
        let tmp_dir = TempDir::default();
        set_tabby_root(tmp_dir.to_path_buf());

        tokio::fs::create_dir_all(tabby_common::path::models_dir())
            .await
            .unwrap();

        tokio::fs::write(
            tabby_common::path::models_dir().join("testfile"),
            "0".repeat(1000).as_bytes(),
        )
        .await
        .unwrap();

        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_analytic_service(db);

        let disk_usage = service.disk_usage_stats().await.unwrap();

        assert_eq!(disk_usage.events.size_kb, 0.0);
        assert_eq!(disk_usage.indexed_repositories.size_kb, 0.0);
        assert_eq!(disk_usage.database.size_kb, 0.0);
        assert_eq!(disk_usage.models.size_kb, 1.0);
    }
}
