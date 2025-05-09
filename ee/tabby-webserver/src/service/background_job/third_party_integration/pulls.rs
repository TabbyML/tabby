use std::pin::Pin;

use anyhow::{anyhow, Result};
use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::{stream::BoxStream, Stream};
use gitlab::api::{projects::merge_requests::MergeRequests as ProjectMergeRequests, Pagination};
use octocrab::{
    models::{pulls::PullRequest, IssueState},
    Octocrab,
};
use serde::Deserialize;
use tabby_index::public::{
    StructuredDoc, StructuredDocFields, StructuredDocPullDocumentFields, StructuredDocState,
};
use tracing::debug;

use super::error::octocrab_error_message;
use crate::service::create_gitlab_client;

#[derive(Deserialize)]
pub struct GitlabMergeRequest {
    pub title: String,
    pub author: GitlabAuthor,
    pub description: Option<String>,
    pub web_url: String,
    pub updated_at: DateTime<Utc>,
    pub state: String,
    pub merged_at: Option<DateTime<Utc>>,
}

#[derive(Deserialize)]
pub struct GitlabAuthor {
    pub public_email: Option<String>,
}

// FIXME(kweizh): we can only get StructuredDoc id after constructing the StructuredDoc
// but we need to pass the id to the StructuredDocState
// so we need to refactor the id() method in StructuredDoc
fn github_pull_id(pull: &PullRequest) -> String {
    pull.html_url
        .clone()
        .map(|url| url.to_string())
        .unwrap_or_else(|| pull.url.clone())
}

fn gitlab_merge_request_id(mr: &GitlabMergeRequest) -> String {
    mr.web_url.clone()
}

pub enum Pull {
    GitHub(PullRequest),
    GitLab(GitlabMergeRequest),
}

pub async fn list_pull_states(
    integration_kind: &tabby_schema::integration::IntegrationKind,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<BoxStream<'static, (Pull, StructuredDocState)>> {
    match integration_kind {
        tabby_schema::integration::IntegrationKind::Github
        | tabby_schema::integration::IntegrationKind::GithubSelfHosted => {
            list_github_pull_states(api_base, full_name, access_token).await
        }
        tabby_schema::integration::IntegrationKind::Gitlab
        | tabby_schema::integration::IntegrationKind::GitlabSelfHosted => {
            list_gitlab_merge_request_states(api_base, full_name, access_token).await
        }
    }
}

async fn list_github_pull_states(
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<Pin<Box<dyn Stream<Item = (Pull, StructuredDocState)> + Send>>> {
    let octocrab = Octocrab::builder()
        .personal_token(access_token.to_string())
        .base_uri(api_base)?
        .build()?;

    let (owner, repo) = full_name
        .split_once('/')
        .ok_or_else(|| anyhow!("Invalid repository name"))?;

    let owner = owner.to_owned();
    let repo = repo.to_owned();
    let s = stream! {
        let mut page = 1u32;
        loop {
            let response = match octocrab
                .pulls(&owner, &repo)
                .list()
                .state(octocrab::params::State::All)
                .page(page)
                .send()
                .await {
                    Ok(x) => x,
                    Err(e) => {
                        logkit::error!("Failed to fetch pull requests: {}", octocrab_error_message(e));
                        break;
                    }
            };

            let pages = response.number_of_pages().unwrap_or_default();

            for pull in response.items {
                let id = github_pull_id(&pull);
                let updated_at = pull.updated_at.unwrap_or_else(chrono::Utc::now);

                // skip closed but not merged pulls
                if let Some(state) = &pull.state {
                    if *state == IssueState::Closed && pull.merged_at.is_none() {
                        yield (Pull::GitHub(pull), StructuredDocState{
                            id,
                            updated_at,
                            deleted: true,
                        });
                        continue;
                    }
                }

                yield (Pull::GitHub(pull), StructuredDocState{
                    id,
                    updated_at,
                    deleted: false,
                });
            }

            page += 1;
            if page > pages {
                break;
            }
        }
    };

    Ok(Box::pin(s))
}

async fn list_gitlab_merge_request_states(
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<Pin<Box<dyn Stream<Item = (Pull, StructuredDocState)> + Send>>> {
    let gitlab_client = create_gitlab_client(api_base, access_token).await?;
    let project_id = full_name.to_owned();

    let result_stream = stream! {
        let endpoint = match ProjectMergeRequests::builder().project(project_id.as_str()).build(){
            Ok(ep) => ep,
            Err(e) => {
                logkit::error!(
                    "Failed to build GitLab MRs endpoint for project {}: {}. Aborting stream.",
                    project_id,
                    e
                );
                return;
            }
        };

        // Get an asynchronous stream of merge requests.
        // The `gitlab_client` already handles pagination.
        let mrs = gitlab::api::paged(endpoint, Pagination::All).into_iter(&gitlab_client).into_async();
        for await mr in mrs {
            match mr {
                Ok(mr) => {
                    let id = gitlab_merge_request_id(&mr);
                    let updated_at = mr.updated_at;
                    // A merge request is considered "deleted" for indexing purposes
                    // if it's in a "closed" state AND was not merged.
                    let deleted = mr.state.as_str() == "closed" && mr.merged_at.is_none();
                    yield (
                        Pull::GitLab(mr),
                        StructuredDocState {
                            id,
                            updated_at,
                            deleted,
                        },
                    );
                }
                Err(e) => {
                    logkit::error!(
                        "Error fetching a page of GitLab merge requests for project {}: {}. Stopping further MR processing for this project.",
                        project_id,
                        e
                    );
                    break;
                }
            }
        }
    };

    Ok(Box::pin(result_stream))
}

pub async fn get_pull_doc(
    source_id: &str,
    pull: Pull,
    integration_kind: &tabby_schema::integration::IntegrationKind,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<StructuredDoc> {
    match integration_kind {
        tabby_schema::integration::IntegrationKind::Github
        | tabby_schema::integration::IntegrationKind::GithubSelfHosted => {
            if let Pull::GitHub(p) = pull {
                get_github_pull_doc(source_id, p, api_base, full_name, access_token).await
            } else {
                Err(anyhow!("Mismatched pull request type"))
            }
        }
        tabby_schema::integration::IntegrationKind::Gitlab
        | tabby_schema::integration::IntegrationKind::GitlabSelfHosted => {
            if let Pull::GitLab(mr) = pull {
                get_gitlab_merge_request_doc(source_id, mr).await
            } else {
                Err(anyhow!("Mismatched merge request type"))
            }
        }
    }
}

async fn get_github_pull_doc(
    source_id: &str,
    pull: PullRequest,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<StructuredDoc> {
    let octocrab = Octocrab::builder()
        .personal_token(access_token.to_string())
        .base_uri(api_base)?
        .build()?;

    let author = pull.user.as_ref().map(|user| user.login.clone());
    let email = if let Some(author) = author {
        match octocrab.users(&author).profile().await {
            Ok(profile) => profile.email,
            Err(e) => {
                debug!(
                    "Failed to fetch user profile for {}: {}",
                    author,
                    octocrab_error_message(e)
                );
                None
            }
        }
    } else {
        None
    };

    let url = pull
        .html_url
        .clone()
        .map(|url| url.to_string())
        .unwrap_or_else(|| pull.url.clone());

    // Fetch the diff only if the size of the diff is less than 1MB,
    // assuming 80 characters per line at most, and 32 at average,
    // so the number of changed lines is fewer than 32,000,
    //
    // When there are more than 300 files,
    // we must utilize the `List pull requests files` API to retrieve the diff,
    // or we will get a 406 status code.
    let diff = if pull.additions.unwrap_or_default() + pull.deletions.unwrap_or_default() < 32000
        && pull.changed_files.unwrap_or_default() < 300
    {
        let (owner, repo) = full_name
            .split_once('/')
            .ok_or_else(|| anyhow!("Invalid repository name"))?;

        match octocrab.pulls(owner, repo).get_diff(pull.number).await {
            Ok(diff) => Some(diff),
            Err(e) => {
                if let octocrab::Error::GitHub { source, .. } = &e {
                    // in most cases, GitHub API does not set the changed_files,
                    // so we need to handle the 406 status code here.
                    if source.status_code == 406 {
                        None
                    } else {
                        return Err(anyhow!(
                            "Failed to fetch pull request diff for {}: {}",
                            url,
                            octocrab_error_message(e)
                        ));
                    }
                } else {
                    return Err(anyhow!(
                        "Failed to fetch pull request diff for {}: {}",
                        url,
                        octocrab_error_message(e)
                    ));
                }
            }
        }
    } else {
        None
    };

    Ok(StructuredDoc {
        source_id: source_id.to_string(),
        fields: StructuredDocFields::Pull(StructuredDocPullDocumentFields {
            link: url,
            title: pull.title.clone().unwrap_or_default(),
            author_email: email.clone(),
            body: pull.body.clone().unwrap_or_default(),
            merged: pull.merged_at.is_some(),
            diff,
        }),
    })
}

async fn get_gitlab_merge_request_doc(
    source_id: &str,
    mr: GitlabMergeRequest,
) -> Result<StructuredDoc> {
    Ok(StructuredDoc {
        source_id: source_id.to_string(),
        fields: StructuredDocFields::Pull(StructuredDocPullDocumentFields {
            link: mr.web_url.clone(),
            title: mr.title.clone(),
            author_email: mr.author.public_email.clone(),
            body: mr.description.clone().unwrap_or_default(),
            merged: mr.merged_at.is_some(),
            diff: None,
        }),
    })
}
