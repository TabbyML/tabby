use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tabby_schema::context::ContextService;

use super::helper::Job;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DbMaintainanceJob;

impl Job for DbMaintainanceJob {
    const NAME: &'static str = "db_maintainance";
}

impl DbMaintainanceJob {
    pub async fn cron(
        _now: DateTime<Utc>,
        context: Arc<dyn ContextService>,
        db: DbConn,
    ) -> tabby_schema::Result<()> {
        db.delete_expired_token().await?;
        db.delete_expired_password_resets().await?;
        db.delete_expired_ephemeral_threads().await?;

        // Read all active sources
        let active_source_ids = context
            .read(None)
            .await?
            .sources
            .into_iter()
            .map(|x| x.source_id())
            .collect::<Vec<_>>();

        db.delete_unused_source_id_read_access_policy(&active_source_ids)
            .await?;
        Ok(())
    }
}
