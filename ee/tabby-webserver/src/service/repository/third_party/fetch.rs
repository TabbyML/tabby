use octocrab::GitHubError;
use tabby_schema::integration::IntegrationKind;

use self::{github::fetch_all_github_repos, gitlab::fetch_all_gitlab_repos};

mod github;
mod gitlab;

pub struct RepositoryInfo {
    pub name: String,
    pub git_url: String,
    pub vendor_id: String,
}

pub async fn fetch_all_repos(
    kind: IntegrationKind,
    access_token: &str,
    api_base: &str,
) -> Result<Vec<RepositoryInfo>, (anyhow::Error, bool)> {
    match kind {
        IntegrationKind::Github | IntegrationKind::GithubSelfHosted => {
            match fetch_all_github_repos(access_token, api_base).await {
                Ok(repos) => Ok(repos),
                Err(octocrab::Error::GitHub {
                    source: source @ GitHubError { .. },
                    ..
                }) if source.status_code.is_client_error() => Err((source.into(), true)),
                Err(e) => Err((e.into(), false)),
            }
        }
        IntegrationKind::Gitlab | IntegrationKind::GitlabSelfHosted => {
            match fetch_all_gitlab_repos(access_token, api_base).await {
                Ok(repos) => Ok(repos),
                Err(e) => {
                    let client_error = e.is_client_error();
                    Err((e.into(), client_error))
                }
            }
        }
    }
}
