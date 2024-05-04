use anyhow::{bail, Result};
use apalis::{
    prelude::{Data, Job, Monitor, Storage, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use gitlab::{
    api::{projects::Projects, ApiError, AsyncQuery, Pagination},
    GitlabBuilder,
};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tracing::debug;

use super::{
    ceprintln, cprintln,
    helper::{BasicJob, CronJob, JobLogger},
};

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

impl CronJob for SyncGitlabJob {
    const SCHEDULE: &'static str = "@hourly";
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
        let monitor = monitor
            .register(Self::basic_worker(storage.clone(), db.clone()).build_fn(Self::run))
            .register(
                Self::cron_worker(db.clone())
                    .data(storage.clone())
                    .build_fn(Self::cron),
            );

        (storage, monitor)
    }
}

async fn refresh_repositories_for_provider(
    logger: JobLogger,
    db: DbConn,
    provider_id: i64,
) -> Result<()> {
    let provider = db.get_gitlab_provider(provider_id).await?;
    cprintln!(
        logger,
        "Refreshing repositories for provider: {}",
        provider.display_name
    );

    let Some(access_token) = provider.access_token else {
        bail!(
            "GitLab provider {} does not have an access token",
            provider.display_name
        );
    };
    let start = Utc::now();
    let repos = match fetch_all_repos(&access_token).await {
        Ok(repos) => repos,
        Err(e) if e.is_client_error() => {
            db.update_gitlab_provider_sync_status(provider_id, false)
                .await?;
            ceprintln!(
                logger,
                "GitLab credentials for provider {} are expired or invalid",
                provider.display_name
            );
            return Err(e.into());
        }
        Err(e) => {
            ceprintln!(logger, "Failed to fetch repositories from gitlab: {e}");
            return Err(e.into());
        }
    };
    for repo in repos {
        cprintln!(logger, "importing: {}", &repo.path_with_namespace);
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

    db.update_gitlab_provider_sync_status(provider_id, true)
        .await?;
    let num_removed = db
        .delete_outdated_gitlab_repositories(provider_id, start.into())
        .await?;
    cprintln!(logger, "Removed {} outdated repositories", num_removed);
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

async fn fetch_all_repos(access_token: &str) -> Result<Vec<Repository>, GitlabError> {
    let gitlab = GitlabBuilder::new("gitlab.com", access_token)
        .build_async()
        .await?;
    Ok(gitlab::api::paged(
        Projects::builder().membership(true).build()?,
        Pagination::All,
    )
    .query_async(&gitlab)
    .await?)
}
