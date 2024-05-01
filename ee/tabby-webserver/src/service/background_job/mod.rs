mod db;
mod github;
mod gitlab;
mod logger;
mod scheduler;

use std::sync::Arc;

use apalis::{
    prelude::{Monitor, Storage},
    sqlite::{SqlitePool, SqliteStorage},
};
use async_trait::async_trait;
use tabby_db::DbConn;

use self::{
    db::DbMaintainanceJob, github::SyncGithubJob, gitlab::SyncGitlabJob, scheduler::SchedulerJob,
};
use crate::path::job_queue;

#[async_trait]
pub trait BackgroundJob: Send + Sync {
    async fn trigger_sync_github(&self, provider_id: i64);
    async fn trigger_sync_gitlab(&self, provider_id: i64);
}

struct BackgroundJobImpl {
    gitlab: SqliteStorage<SyncGitlabJob>,
    github: SqliteStorage<SyncGithubJob>,
}

pub async fn create(db: DbConn, local_port: u16) -> Arc<dyn BackgroundJob> {
    let path = format!("sqlite://{}?mode=rwc", job_queue().display());
    let pool = SqlitePool::connect(&path)
        .await
        .expect("unable to create sqlite pool");
    SqliteStorage::setup(&pool)
        .await
        .expect("unable to run migrations for sqlite");

    let monitor = Monitor::new();
    let monitor = DbMaintainanceJob::register(monitor, db.clone());
    let monitor = SchedulerJob::register(monitor, db.clone(), local_port);
    let (gitlab, monitor) = SyncGitlabJob::register(monitor, pool.clone(), db.clone());
    let (github, monitor) = SyncGithubJob::register(monitor, pool.clone(), db.clone());

    tokio::spawn(async move {
        monitor.run().await.expect("failed to start worker");
    });

    Arc::new(BackgroundJobImpl { gitlab, github })
}

struct FakeBackgroundJob;

#[async_trait]
impl BackgroundJob for FakeBackgroundJob {
    async fn trigger_sync_github(&self, _provider_id: i64) {}
    async fn trigger_sync_gitlab(&self, _provider_id: i64) {}
}

#[cfg(test)]
pub fn create_fake() -> Arc<dyn BackgroundJob> {
    Arc::new(FakeBackgroundJob)
}

#[async_trait]
impl BackgroundJob for BackgroundJobImpl {
    async fn trigger_sync_github(&self, provider_id: i64) {
        self.github
            .clone()
            .push(SyncGithubJob::new(provider_id))
            .await
            .expect("unable to push job");
    }

    async fn trigger_sync_gitlab(&self, provider_id: i64) {
        self.gitlab
            .clone()
            .push(SyncGitlabJob::new(provider_id))
            .await
            .expect("unable to push job");
    }
}
