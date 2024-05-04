mod db;
mod github;
mod gitlab;
mod helper;
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
use crate::path::job_db_file;

#[async_trait]
pub trait BackgroundJob: Send + Sync {
    async fn trigger_scheduler(&self);
    async fn trigger_sync_github(&self, provider_id: i64);
    async fn trigger_sync_gitlab(&self, provider_id: i64);
}

struct BackgroundJobImpl {
    scheduler: SqliteStorage<SchedulerJob>,
    gitlab: SqliteStorage<SyncGitlabJob>,
    github: SqliteStorage<SyncGithubJob>,
}

pub async fn create(db: DbConn, local_port: u16) -> Arc<dyn BackgroundJob> {
    let path = format!("sqlite://{}?mode=rwc", job_db_file().display());
    let pool = SqlitePool::connect(&path)
        .await
        .expect("unable to create sqlite pool");
    SqliteStorage::setup(&pool)
        .await
        .expect("unable to run migrations for sqlite");

    let monitor = Monitor::new();
    let monitor = DbMaintainanceJob::register(monitor, db.clone());
    let (scheduler, monitor) =
        SchedulerJob::register(monitor, pool.clone(), db.clone(), local_port);
    let (gitlab, monitor) = SyncGitlabJob::register(monitor, pool.clone(), db.clone());
    let (github, monitor) = SyncGithubJob::register(monitor, pool.clone(), db.clone());

    tokio::spawn(async move {
        monitor.run().await.expect("failed to start worker");
    });

    Arc::new(BackgroundJobImpl {
        scheduler,
        gitlab,
        github,
    })
}

struct FakeBackgroundJob;

#[async_trait]
impl BackgroundJob for FakeBackgroundJob {
    async fn trigger_scheduler(&self) {}
    async fn trigger_sync_github(&self, _provider_id: i64) {}
    async fn trigger_sync_gitlab(&self, _provider_id: i64) {}
}

#[cfg(test)]
pub fn create_fake() -> Arc<dyn BackgroundJob> {
    Arc::new(FakeBackgroundJob)
}

#[async_trait]
impl BackgroundJob for BackgroundJobImpl {
    async fn trigger_scheduler(&self) {
        self.scheduler
            .clone()
            .push(SchedulerJob)
            .await
            .expect("unable to push job");
    }

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

macro_rules! ceprintln {
    ($ctx:expr, $($params:tt)+) => {
        {
            tracing::warn!($($params)+);
            $ctx.r#internal_eprintln(format!($($params)+)).await;
        }
    }
}

macro_rules! cprintln {
    ($ctx:expr, $($params:tt)+) => {
        {
            tracing::debug!($($params)+);
            $ctx.r#internal_println(format!($($params)+)).await;
        }
    }
}

use ceprintln;
use cprintln;
