mod db;
mod github;
mod gitlab;
mod scheduler;

use std::sync::Arc;

use tokio_cron_scheduler::{Job, JobScheduler};

use crate::schema::{
    auth::AuthenticationService, github_repository_provider::GithubRepositoryProviderService,
    job::JobService, worker::WorkerService,
};

async fn new_job_scheduler(jobs: Vec<Job>) -> anyhow::Result<JobScheduler> {
    let scheduler = JobScheduler::new().await?;
    for job in jobs {
        scheduler.add(job).await?;
    }
    scheduler.start().await?;
    Ok(scheduler)
}

pub async fn run_cron(
    auth: Arc<dyn AuthenticationService>,
    job: Arc<dyn JobService>,
    worker: Arc<dyn WorkerService>,
    github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
    local_port: u16,
) {
    let mut jobs = vec![];

    let job1 = db::refresh_token_job(auth.clone())
        .await
        .expect("failed to create refresh token cleanup job");
    jobs.push(job1);

    let job2 = db::password_reset_job(auth)
        .await
        .expect("failed to create password reset token cleanup job");
    jobs.push(job2);

    let job3 = scheduler::scheduler_job(job.clone(), worker, local_port)
        .await
        .expect("failed to create scheduler job");
    jobs.push(job3);

    let job4 = db::job_cleanup(job)
        .await
        .expect("failed to create stale job runs cleanup job");
    jobs.push(job4);

    let job5 = db::update_integrated_github_repositories_job(github_repository_provider)
        .await
        .expect("Failed to create github repository refresh job");
    jobs.push(job5);

    new_job_scheduler(jobs)
        .await
        .expect("failed to start job scheduler");
}
