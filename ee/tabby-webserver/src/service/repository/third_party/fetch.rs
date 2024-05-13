use gitlab::{
    api::{projects::Projects, AsyncQuery, Pagination},
    GitlabBuilder,
};
use octocrab::{GitHubError, Octocrab};
use tabby_schema::integration::IntegrationKind;

pub struct RepositoryInfo {
    pub name: String,
    pub git_url: String,
    pub vendor_id: String,
}

mod gitlab_types {
    use gitlab::api::ApiError;
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
}

pub async fn fetch_all_repos(
    kind: IntegrationKind,
    access_token: &str,
) -> Result<Vec<RepositoryInfo>, (anyhow::Error, bool)> {
    match kind {
        IntegrationKind::Github => match fetch_all_github_repos(access_token).await {
            Ok(repos) => Ok(repos),
            Err(octocrab::Error::GitHub {
                source: source @ GitHubError { .. },
                ..
            }) if source.status_code.is_client_error() => Err((source.into(), true)),
            Err(e) => Err((e.into(), false)),
        },
        IntegrationKind::Gitlab => match fetch_all_gitlab_repos(access_token).await {
            Ok(repos) => Ok(repos),
            Err(e) => {
                let client_error = e.is_client_error();
                Err((e.into(), client_error))
            }
        },
    }
}

async fn fetch_all_gitlab_repos(
    access_token: &str,
) -> Result<Vec<RepositoryInfo>, gitlab_types::GitlabError> {
    let gitlab = GitlabBuilder::new("gitlab.com", access_token)
        .build_async()
        .await?;
    let repos: Vec<gitlab_types::GitlabRepository> = gitlab::api::paged(
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

async fn fetch_all_github_repos(
    access_token: &str,
) -> Result<Vec<RepositoryInfo>, octocrab::Error> {
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
        repos.extend(response.items.into_iter().filter_map(|repo| {
            Some(RepositoryInfo {
                name: repo.full_name.unwrap_or(repo.name),
                git_url: repo.html_url?.to_string(),
                vendor_id: repo.id.to_string(),
            })
        }));

        page += 1;
        if page > pages {
            break;
        }
    }
    Ok(repos)
}
