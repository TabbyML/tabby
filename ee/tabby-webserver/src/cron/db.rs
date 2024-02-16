//! db maintenance jobs

use std::sync::Arc;

use anyhow::Result;
use futures::Future;
use tokio_cron_scheduler::Job;
use tracing::error;

use crate::schema::auth::AuthenticationService;

async fn auth_job<F>(
    auth: Arc<dyn AuthenticationService>,
    job: fn(Arc<dyn AuthenticationService>) -> F,
) -> Result<Job>
where
    F: Future<Output = Result<()>> + 'static + Send,
{
    // job is run every 2 hours
    let job = Job::new_async("0 0 1/2 * * * *", move |_, _| {
        let auth = auth.clone();
        Box::pin(async move {
            let res = job(auth.clone()).await;
            if let Err(e) = res {
                error!("failed to delete expired token: {}", e);
            }
        })
    })?;

    Ok(job)
}

pub async fn refresh_token_job(auth: Arc<dyn AuthenticationService>) -> Result<Job> {
    auth_job(
        auth,
        |auth| async move { auth.delete_expired_token().await },
    )
    .await
}

pub async fn password_reset_job(auth: Arc<dyn AuthenticationService>) -> Result<Job> {
    auth_job(auth, |auth| async move {
        auth.delete_expired_password_resets().await
    })
    .await
}
