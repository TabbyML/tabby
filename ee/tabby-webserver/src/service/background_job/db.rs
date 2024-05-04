use apalis::{
    prelude::{Data, Job, Monitor, WorkerFactoryFn},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;

use super::{
    cprintln,
    helper::{CronJob, JobLogger},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DbMaintainanceJob;

impl Job for DbMaintainanceJob {
    const NAME: &'static str = "db_maintainance";
}

impl CronJob for DbMaintainanceJob {
    const SCHEDULE: &'static str = "@hourly";
}

impl DbMaintainanceJob {
    async fn cron(
        _now: DateTime<Utc>,
        logger: Data<JobLogger>,
        db: Data<DbConn>,
    ) -> tabby_schema::Result<()> {
        cprintln!(logger, "Running db maintainance job");
        db.delete_expired_token().await?;
        db.delete_expired_password_resets().await?;
        Ok(())
    }

    pub fn register(monitor: Monitor<TokioExecutor>, db: DbConn) -> Monitor<TokioExecutor> {
        monitor.register(Self::cron_worker(db.clone()).build_fn(DbMaintainanceJob::cron))
    }
}
