//! db maintenance jobs

use std::sync::Arc;

use anyhow::Result;
use tokio_cron_scheduler::Job;
use tracing::error;

use crate::schema::auth::AuthenticationService;

pub async fn refresh_token_job(auth: Arc<dyn AuthenticationService>) -> Result<Job> {
    // job is run every 2 hours
    let job = Job::new_async("0 0 1/2 * * * *", move |_, _| {
        let auth = auth.clone();
        Box::pin(async move {
            let res = auth.delete_expired_token().await;
            if let Err(e) = res {
                error!("failed to delete expired token: {}", e);
            }
        })
    })?;

    Ok(job)
}
