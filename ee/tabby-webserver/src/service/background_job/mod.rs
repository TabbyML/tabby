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
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tabby_db::DbConn;

use self::{
    db::DbMaintainanceJob, github::SyncGithubJob, gitlab::SyncGitlabJob, scheduler::SchedulerJob,
};
use crate::path::job_db_file;

pub enum BackgroundJobEvent {
    Scheduler(RepositoryConfig),
    SyncGithub(i64),
    SyncGitlab(i64),
}

struct BackgroundJobImpl {
    scheduler: SqliteStorage<SchedulerJob>,
    gitlab: SqliteStorage<SyncGitlabJob>,
    github: SqliteStorage<SyncGithubJob>,
}

pub async fn start(
    db: DbConn,
    repository_access: Arc<dyn RepositoryAccess>,
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<BackgroundJobEvent>,
) {
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
        SchedulerJob::register(monitor, pool.clone(), db.clone(), repository_access);
    let (gitlab, monitor) = SyncGitlabJob::register(monitor, pool.clone(), db.clone());
    let (github, monitor) = SyncGithubJob::register(monitor, pool.clone(), db.clone());

    tokio::spawn(async move {
        monitor.run().await.expect("failed to start worker");
    });

    tokio::spawn(async move {
        let mut background_job = BackgroundJobImpl {
            scheduler,
            gitlab,
            github,
        };

        while let Some(event) = receiver.recv().await {
            background_job.on_event_publish(event).await;
        }
    });
}

impl BackgroundJobImpl {
    async fn trigger_scheduler(&self, repository: RepositoryConfig) {
        self.scheduler
            .clone()
            .push(SchedulerJob::new(repository))
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

    async fn on_event_publish(&mut self, event: BackgroundJobEvent) {
        match event {
            BackgroundJobEvent::Scheduler(repository) => self.trigger_scheduler(repository).await,
            BackgroundJobEvent::SyncGithub(provider_id) => {
                self.trigger_sync_github(provider_id).await
            }
            BackgroundJobEvent::SyncGitlab(provider_id) => {
                self.trigger_sync_gitlab(provider_id).await
            }
        }
    }
}

macro_rules! ceprintln {
    ($ctx:expr, $($params:tt)+) => {
        {
            tracing::debug!($($params)+);
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
