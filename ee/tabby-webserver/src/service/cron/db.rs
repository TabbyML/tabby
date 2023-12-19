//! db maintenance jobs

use anyhow::Result;
use tokio_cron_scheduler::Job;
use tracing::error;

use crate::service::db::DbConn;

pub async fn refresh_token_job(db_conn: DbConn) -> Result<Job> {
    // job is run every 2 hours
    let job = Job::new_async("0 0 1/2 * * * *", move |_, _| {
        let db_conn = db_conn.clone();
        Box::pin(async move {
            let res = db_conn.delete_expired_token().await;
            if let Err(e) = res {
                error!("failed to delete expired token: {}", e);
            }
        })
    })?;

    Ok(job)
}
