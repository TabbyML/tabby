use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;

use super::helper::Job;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DbMaintainanceJob;

impl Job for DbMaintainanceJob {
    const NAME: &'static str = "db_maintainance";
}

impl DbMaintainanceJob {
    pub async fn cron(_now: DateTime<Utc>, db: DbConn) -> tabby_schema::Result<()> {
        db.delete_expired_token().await?;
        db.delete_expired_password_resets().await?;
        Ok(())
    }
}
