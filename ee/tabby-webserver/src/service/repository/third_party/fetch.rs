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
) -> Result<Vec<RepositoryInfo>, anyhow::Error> {
    match kind {
        IntegrationKind::Github | IntegrationKind::GithubSelfHosted => {
            Ok(fetch_all_github_repos(access_token, api_base).await?)
        }
        IntegrationKind::Gitlab | IntegrationKind::GitlabSelfHosted => {
            Ok(fetch_all_gitlab_repos(access_token, api_base).await?)
        }
    }
}
