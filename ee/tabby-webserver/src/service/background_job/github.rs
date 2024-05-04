use anyhow::{bail, Result};
use apalis::{
    prelude::{Data, Job, Monitor, Storage, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use octocrab::{models::Repository, GitHubError, Octocrab};
use serde::{Deserialize, Serialize};
use tabby_db::{DbConn, GithubRepositoryProviderDAO};
use tracing::debug;

use super::{
    ceprintln, cprintln,
    helper::{BasicJob, CronJob, JobLogger},
};

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

impl CronJob for SyncGithubJob {
    const SCHEDULE: &'static str = "@hourly";
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

async fn refresh_repositories_for_provider(logger: JobLogger, db: DbConn, provider_id: i64) -> Result<()> {
    let provider = db.get_github_provider(provider_id).await?;
    cprintln!(
        logger,
        "Refreshing repositories for provider: {}",
        provider.display_name
    );

    let Some(access_token) = provider.access_token else {
        bail!("Github provider {} does not have an access token", provider.display_name);
    };
    let start = Utc::now();
    let repos = match fetch_all_repos(&access_token).await {
        Ok(repos) => repos,
        Err(octocrab::Error::GitHub {
            source: source @ GitHubError { .. },
            ..
        }) if source.status_code.is_client_error() => {
            db.update_github_provider_sync_status(provider_id, false)
                .await?;
            ceprintln!(
                logger,
                "GitHub credentials for provider {} are expired or invalid",
                provider.display_name
            );
            return Err(source.into());
        }
        Err(e) => {
            ceprintln!(logger, "Failed to fetch repositories from github: {e}");
            return Err(e.into());
        }
    };
    for repo in repos {
        cprintln!(
            logger,
            "importing: {}",
            repo.full_name.as_deref().unwrap_or(&repo.name)
        );

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
    db.delete_outdated_github_repositories(provider_id, start.into())
        .await?;
    Ok(())
}

// FIXME(wsxiaoys): Convert to async stream
async fn fetch_all_repos(
    access_token: &str
) -> Result<Vec<Repository>, octocrab::Error> {
    let octocrab = Octocrab::builder()
        .user_access_token(access_token.to_string())
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
