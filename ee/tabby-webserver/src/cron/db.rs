//! db maintenance jobs

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use futures::Future;
use tokio_cron_scheduler::Job;
use tracing::{debug, error};

use super::{github, gitlab};
use crate::schema::{
    auth::AuthenticationService, github_repository_provider::GithubRepositoryProviderService,
    gitlab_repository_provider::GitlabRepositoryProviderService, job::JobService,
};

const EVERY_TWO_HOURS: &str = "0 0 1/2 * * * *";
const EVERY_TEN_MINUTES: &str = "0 1/10 * * * *";

async fn service_job<F, S>(
    frequency: &'static str,
    service: Arc<S>,
    job: fn(Arc<S>) -> F,
) -> Result<Job>
where
    F: Future<Output = Result<()>> + 'static + Send,
    S: Send + Sync + 'static + ?Sized,
{
    let job = Job::new_async(frequency, move |_, _| {
        let auth = service.clone();
        Box::pin(async move {
            let res = job(auth.clone()).await;
            if let Err(e) = res {
                error!("failed to run cleanup job: {}", e);
            }
        })
    })?;

    Ok(job)
}

pub async fn refresh_token_job(auth: Arc<dyn AuthenticationService>) -> Result<Job> {
    service_job(EVERY_TWO_HOURS, auth, |auth| async move {
        Ok(auth.delete_expired_token().await?)
    })
    .await
}

pub async fn password_reset_job(auth: Arc<dyn AuthenticationService>) -> Result<Job> {
    service_job(EVERY_TWO_HOURS, auth, |auth| async move {
        Ok(auth.delete_expired_password_resets().await?)
    })
    .await
}

pub async fn update_integrated_github_repositories_job(
    github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
) -> Result<Job> {
    service_job(
        EVERY_TEN_MINUTES,
        github_repository_provider,
        |github_repository_provider| async move {
            debug!("Syncing github repositories...");
            github::refresh_all_repositories(github_repository_provider).await
        },
    )
    .await
}

pub async fn update_integrated_gitlab_repositories_job(
    gitlab_repository_provider: Arc<dyn GitlabRepositoryProviderService>,
) -> Result<Job> {
    service_job(
        EVERY_TEN_MINUTES,
        gitlab_repository_provider,
        |gitlab_repository_provider| async move {
            debug!("Syncing gitlab repositories...");
            gitlab::refresh_all_repositories(gitlab_repository_provider).await
        },
    )
    .await
}

pub async fn job_cleanup(jobs: Arc<dyn JobService>) -> Result<Job> {
    let job_res = Job::new_one_shot_async(Duration::from_secs(0), move |_, _| {
        let jobs = jobs.clone();
        Box::pin(async move {
            if let Err(e) = jobs.cleanup().await {
                error!("failed to finalize stale job runs: {e}");
            }
        })
    });
    Ok(job_res?)
}
