use anyhow::Result;
use gitlab::api::{projects::Projects, AsyncQuery, Pagination};
use serde::Deserialize;

use super::RepositoryInfo;
use crate::service::create_gitlab_client;

#[derive(Deserialize)]
pub struct GitlabRepository {
    pub id: u128,
    pub path_with_namespace: String,
    pub http_url_to_repo: String,
}

#[derive(thiserror::Error, Debug)]
pub enum GitlabError {
    #[error(transparent)]
    Rest(#[from] gitlab::api::ApiError<gitlab::RestError>),
    #[error(transparent)]
    Gitlab(#[from] gitlab::GitlabError),
    #[error(transparent)]
    Projects(#[from] gitlab::api::projects::ProjectsBuilderError),
}

pub async fn fetch_all_gitlab_repos(
    access_token: &str,
    api_base: &str,
) -> Result<Vec<RepositoryInfo>> {
    let gitlab = create_gitlab_client(api_base, access_token).await?;
    let repos: Vec<GitlabRepository> = gitlab::api::paged(
        Projects::builder().membership(true).build()?,
        Pagination::All,
    )
    .query_async(&gitlab)
    .await?;

    Ok(repos
        .into_iter()
        .map(|repo| RepositoryInfo {
            name: repo.path_with_namespace,
            git_url: repo.http_url_to_repo,
            vendor_id: repo.id.to_string(),
        })
        .collect())
}
