use std::str::FromStr;

use anyhow::Result;
use apalis::{
    cron::{CronStream, Schedule},
    prelude::{Data, Job, Monitor, Storage, WorkerBuilder, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use gitlab::{
    api::{projects::Projects, ApiError, AsyncQuery, Pagination},
    GitlabBuilder,
};
use serde::{Deserialize, Serialize};
use tabby_db::{DbConn, GitlabRepositoryProviderDAO};
use tower::limit::ConcurrencyLimitLayer;
use tracing::debug;

use super::layer::{JobLogLayer, JobLogger};
use crate::warn_stderr;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SyncGitlabJob {
    provider_id: i64,
}

impl SyncGitlabJob {
    pub fn new(provider_id: i64) -> Self {
        Self { provider_id }
    }
}

impl Job for SyncGitlabJob {
    const NAME: &'static str = "import_gitlab_repositories";
}

impl SyncGitlabJob {
    async fn run(self, logger: Data<JobLogger>, db: Data<DbConn>) -> tabby_schema::Result<()> {
        refresh_repositories_for_provider((*logger).clone(), (*db).clone(), self.provider_id)
            .await?;
        Ok(())
    }

    async fn cron(
        _now: DateTime<Utc>,
        storage: Data<SqliteStorage<SyncGitlabJob>>,
        db: Data<DbConn>,
    ) -> tabby_schema::Result<()> {
        debug!("Syncing all gitlab providers");

        let mut storage = (*storage).clone();
        for provider in db
            .list_gitlab_repository_providers(vec![], None, None, false)
            .await?
        {
            storage
                .push(SyncGitlabJob::new(provider.id))
                .await
                .expect("unable to push job");
        }

        Ok(())
    }

    pub fn register(
        monitor: Monitor<TokioExecutor>,
        pool: SqlitePool,
        db: DbConn,
    ) -> (SqliteStorage<SyncGitlabJob>, Monitor<TokioExecutor>) {
        let storage = SqliteStorage::new(pool);
        let schedule = Schedule::from_str("@hourly").expect("unable to parse cron schedule");
        let monitor = monitor
            .register(
                WorkerBuilder::new(Self::NAME)
                    .with_storage(storage.clone())
                    .layer(ConcurrencyLimitLayer::new(1))
                    .layer(JobLogLayer::new(db.clone(), Self::NAME))
                    .data(db.clone())
                    .build_fn(Self::run),
            )
            .register(
                WorkerBuilder::new(format!("{}-cron", Self::NAME))
                    .stream(CronStream::new(schedule).into_stream())
                    .data(storage.clone())
                    .data(db.clone())
                    .build_fn(Self::cron),
            );

        (storage, monitor)
    }
}

async fn refresh_repositories_for_provider(logger: JobLogger, db: DbConn, id: i64) -> Result<()> {
    let provider = db.get_gitlab_provider(id).await?;
    logger
        .stdout_writeline(format!(
            "Refreshing repositories for provider: {}\n",
            provider.display_name
        ))
        .await;
    let start = Utc::now();
    let repos = match fetch_all_repos(&provider).await {
        Ok(repos) => repos,
        Err(e) if e.is_client_error() => {
            db.update_gitlab_provider_sync_status(id, false).await?;
            warn_stderr!(
                logger,
                "GitLab credentials for provider {} are expired or invalid",
                provider.display_name
            );
            return Err(e.into());
        }
        Err(e) => {
            warn_stderr!(logger, "Failed to fetch repositories from gitlab: {e}");
            return Err(e.into());
        }
    };
    for repo in repos {
        logger
            .stdout_writeline(format!("importing: {}", &repo.path_with_namespace))
            .await;
        let id = repo.id.to_string();
        let url = repo.http_url_to_repo;
        let url = url.strip_suffix(".git").unwrap_or(&url);

        db.upsert_gitlab_provided_repository(
            provider.id,
            id,
            repo.path_with_namespace,
            url.to_string(),
        )
        .await?;
    }

    db.update_gitlab_provided_repository_active(id, true)
        .await?;
    db.delete_outdated_gitlab_repositories(id, start.into())
        .await?;
    Ok(())
}

#[derive(Deserialize)]
struct Repository {
    id: u128,
    path_with_namespace: String,
    http_url_to_repo: String,
}

#[derive(thiserror::Error, Debug)]
enum GitlabError {
    #[error(transparent)]
    Rest(#[from] gitlab::api::ApiError<gitlab::RestError>),
    #[error(transparent)]
    Gitlab(#[from] gitlab::GitlabError),
    #[error(transparent)]
    Projects(#[from] gitlab::api::projects::ProjectsBuilderError),
}

impl GitlabError {
    fn is_client_error(&self) -> bool {
        match self {
            GitlabError::Rest(source)
            | GitlabError::Gitlab(gitlab::GitlabError::Api { source }) => {
                matches!(
                    source,
                    ApiError::Auth { .. }
                        | ApiError::Client {
                            source: gitlab::RestError::AuthError { .. }
                        }
                        | ApiError::Gitlab { .. }
                )
            }
            _ => false,
        }
    }
}

async fn fetch_all_repos(
    provider: &GitlabRepositoryProviderDAO,
) -> Result<Vec<Repository>, GitlabError> {
    let Some(token) = &provider.access_token else {
        return Ok(vec![]);
    };
    let gitlab = GitlabBuilder::new("gitlab.com", token)
        .build_async()
        .await?;
    Ok(gitlab::api::paged(
        Projects::builder().membership(true).build()?,
        Pagination::All,
    )
    .query_async(&gitlab)
    .await?)
}
