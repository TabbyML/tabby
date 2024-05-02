use std::str::FromStr;

use anyhow::Result;
use apalis::{
    cron::{CronStream, Schedule},
    prelude::{Data, Job, Monitor, Storage, WorkerBuilder, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use octocrab::{models::Repository, GitHubError, Octocrab};
use serde::{Deserialize, Serialize};
use tabby_db::{DbConn, GithubRepositoryProviderDAO};
use tower::limit::ConcurrencyLimitLayer;
use tracing::debug;

use super::logger::{JobLogLayer, JobLogger};
use crate::warn_stderr;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SyncGithubJob {
    provider_id: i64,
}

impl SyncGithubJob {
    pub fn new(provider_id: i64) -> Self {
        Self { provider_id }
    }
}

impl Job for SyncGithubJob {
    const NAME: &'static str = "import_github_repositories";
}

impl SyncGithubJob {
    async fn run(self, logger: Data<JobLogger>, db: Data<DbConn>) -> tabby_schema::Result<()> {
        refresh_repositories_for_provider((*logger).clone(), (*db).clone(), self.provider_id)
            .await?;
        Ok(())
    }

    async fn cron(
        _now: DateTime<Utc>,
        storage: Data<SqliteStorage<SyncGithubJob>>,
        db: Data<DbConn>,
    ) -> tabby_schema::Result<()> {
        debug!("Syncing all github providers");

        let mut storage = (*storage).clone();
        for provider in db
            .list_github_repository_providers(vec![], None, None, false)
            .await?
        {
            storage
                .push(SyncGithubJob::new(provider.id))
                .await
                .expect("unable to push job");
        }

        Ok(())
    }

    pub fn register(
        monitor: Monitor<TokioExecutor>,
        pool: SqlitePool,
        db: DbConn,
    ) -> (SqliteStorage<SyncGithubJob>, Monitor<TokioExecutor>) {
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

async fn refresh_repositories_for_provider(
    context: JobLogger,
    db: DbConn,
    provider_id: i64,
) -> Result<()> {
    let provider = db.get_github_provider(provider_id).await?;
    let repos = match fetch_all_repos(&provider).await {
        Ok(repos) => repos,
        Err(octocrab::Error::GitHub {
            source: source @ GitHubError { .. },
            ..
        }) if source.status_code.is_client_error() => {
            db.update_github_provider_sync_status(provider_id, false)
                .await?;
            warn_stderr!(
                context,
                "GitHub credentials for provider {} are expired or invalid",
                provider.display_name
            );
            return Err(source.into());
        }
        Err(e) => {
            warn_stderr!(context, "Failed to fetch repositories from github: {e}");
            return Err(e.into());
        }
    };
    for repo in repos {
        context
            .stdout_writeline(format!(
                "importing: {}",
                repo.full_name.as_deref().unwrap_or(&repo.name)
            ))
            .await;

        let id = repo.id.to_string();
        let Some(url) = repo.git_url else {
            continue;
        };
        let url = url.to_string();
        let url = url
            .strip_prefix("git://")
            .map(|url| format!("https://{url}"))
            .unwrap_or(url);
        let url = url.strip_suffix(".git").unwrap_or(&url);

        db.upsert_github_provided_repository(
            provider_id,
            id,
            repo.full_name.unwrap_or(repo.name),
            url.to_string(),
        )
        .await?;
    }
    db.update_github_provider_sync_status(provider_id, true)
        .await?;

    Ok(())
}

// FIXME(wsxiaoys): Convert to async stream
async fn fetch_all_repos(
    provider: &GithubRepositoryProviderDAO,
) -> Result<Vec<Repository>, octocrab::Error> {
    let Some(token) = &provider.access_token else {
        return Ok(vec![]);
    };
    let octocrab = Octocrab::builder()
        .user_access_token(token.to_string())
        .build()?;

    let mut page = 1;
    let mut repos = vec![];

    loop {
        let response = octocrab
            .current()
            .list_repos_for_authenticated_user()
            .visibility("all")
            .page(page)
            .send()
            .await?;

        let pages = response.number_of_pages().unwrap_or_default() as u8;
        repos.extend(response.items);

        page += 1;
        if page > pages {
            break;
        }
    }
    Ok(repos)
}
