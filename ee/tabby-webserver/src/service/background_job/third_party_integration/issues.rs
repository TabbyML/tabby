use anyhow::{anyhow, Result};
use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::Stream;
use gitlab::api::{issues::ProjectIssues, projects::merge_requests::MergeRequests, AsyncQuery};
use octocrab::Octocrab;
use serde::Deserialize;
use tabby_scheduler::WebDocument;

use crate::service::create_gitlab_client;

pub async fn list_github_issues(
    source_id: &str,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<impl Stream<Item = (DateTime<Utc>, WebDocument)>> {
    let octocrab = Octocrab::builder()
        .personal_token(access_token.to_string())
        .base_uri(api_base)?
        .build()?;

    let (owner, repo) = full_name
        .split_once('/')
        .ok_or_else(|| anyhow!("Invalid repository name"))?;

    let owner = owner.to_owned();
    let repo = repo.to_owned();
    let source_id = source_id.to_owned();
    let s = stream! {
        let mut page = 1u32;
        loop {
            let response = match octocrab
                .issues(&owner, &repo)
                .list()
                .state(octocrab::params::State::All)
                .page(page)
                .send()
                .await {
                    Ok(x) => x,
                    Err(e) => {
                        logkit::error!("Failed to fetch issues: {}", e);
                        break;
                    }
            };

            let pages = response.number_of_pages().unwrap_or_default();

            for issue in response.items {
                let doc = WebDocument {
                    source_id: source_id.to_string(),
                    id: issue.html_url.to_string(),
                    link: issue.html_url.to_string(),
                    title: issue.title,
                    body: issue.body.unwrap_or_default(),
                };
                yield (issue.updated_at, doc);
            }

            page += 1;
            if page > pages {
                break;
            }
        }
    };

    Ok(s)
}

#[derive(Deserialize)]
struct GitlabIssue {
    title: String,
    description: String,
    web_url: String,
    updated_at: DateTime<Utc>,
}

pub async fn list_gitlab_issues(
    source_id: &str,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<impl Stream<Item = (DateTime<Utc>, WebDocument)>> {
    let gitlab = create_gitlab_client(api_base, access_token).await?;

    let source_id = source_id.to_owned();
    let full_name = full_name.to_owned();
    let s = stream! {

        let issues: Vec<GitlabIssue> = match gitlab::api::paged(
            ProjectIssues::builder().project(&full_name).build().expect("Failed to build request"),
            gitlab::api::Pagination::All,
        )
        .query_async(&gitlab)
        .await {
            Ok(x) => x,
            Err(e) => {
                logkit::error!("Failed to fetch issues: {}", e);
                return;
            }
        };

        for issue in issues {
            let doc = WebDocument {
                source_id: source_id.to_owned(),
                id: issue.web_url.clone(),
                link: issue.web_url,
                title: issue.title,
                body: issue.description,
            };
            yield (issue.updated_at, doc);
        }

        let merge_requests: Vec<GitlabIssue> = match gitlab::api::paged(
            MergeRequests::builder().project(&full_name).build().expect("Failed to build request"),
            gitlab::api::Pagination::All,
        )
        .query_async(&gitlab)
        .await {
            Ok(x) => x,
            Err(e) => {
                logkit::error!("Failed to fetch merge requests: {}", e);
                return;
            }
        };

        for merge_request in merge_requests {
            let doc = WebDocument {
                source_id: source_id.to_owned(),
                id: merge_request.web_url.clone(),
                link: merge_request.web_url,
                title: merge_request.title,
                body: merge_request.description,
            };
            yield (merge_request.updated_at, doc);
        }

    };

    Ok(s)
}
