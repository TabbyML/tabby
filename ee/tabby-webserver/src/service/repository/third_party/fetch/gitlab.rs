use super::RepositoryInfo;
use gitlab::{
    api::{projects::Projects, ApiError, AsyncQuery, Pagination},
    GitlabBuilder,
};
use serde::Deserialize;

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

impl GitlabError {
    pub fn is_client_error(&self) -> bool {
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

pub async fn fetch_all_gitlab_repos(
    access_token: &str,
) -> Result<Vec<RepositoryInfo>, GitlabError> {
    let gitlab = GitlabBuilder::new("gitlab.com", access_token)
        .build_async()
        .await?;
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
