use anyhow::{anyhow, Result};
use gitlab::api::{issues::ProjectIssues, projects::merge_requests::MergeRequests, AsyncQuery};
use octocrab::Octocrab;
use serde::Deserialize;
use tabby_scheduler::{DocIndexer, WebDocument};

use crate::service::create_gitlab_client;

pub async fn index_github_issues(
    source_id: &str,
    api_base: &str,
    full_name: &str,
    access_token: &str,
    index: &DocIndexer,
) -> Result<()> {
    let octocrab = Octocrab::builder()
        .personal_token(access_token.to_string())
        .base_uri(api_base)?
        .build()?;

    let (owner, repo) = full_name
        .split_once('/')
        .ok_or_else(|| anyhow!("Invalid repository name"))?;

    let mut page = 1u32;
    let mut issues = vec![];

    loop {
        let response = octocrab
            .issues(owner, repo)
            .list()
            .state(octocrab::params::State::All)
            .page(page)
            .send()
            .await?;

        let pages = response.number_of_pages().unwrap_or_default();
        issues.extend(response.items);
        page += 1;

        if page > pages {
            break;
        }
    }

    for issue in issues {
        let doc = WebDocument {
            source_id: source_id.to_string(),
            id: issue.html_url.to_string(),
            link: issue.html_url.to_string(),
            title: issue.title,
            body: issue.body.unwrap_or_default(),
        };
        index.add(doc).await;
    }

    Ok(())
}

#[derive(Deserialize)]
struct GitlabIssue {
    title: String,
    description: String,
    web_url: String,
}

pub async fn index_gitlab_issues(
    source_id: &str,
    api_base: &str,
    full_name: &str,
    access_token: &str,
    index: &DocIndexer,
) -> Result<()> {
    let gitlab = create_gitlab_client(api_base, access_token).await?;

    let issues: Vec<GitlabIssue> = gitlab::api::paged(
        ProjectIssues::builder().project(full_name).build()?,
        gitlab::api::Pagination::All,
    )
    .query_async(&gitlab)
    .await?;

    for issue in issues {
        let doc = WebDocument {
            source_id: source_id.to_owned(),
            id: issue.web_url.clone(),
            link: issue.web_url,
            title: issue.title,
            body: issue.description,
        };
        index.add(doc).await;
    }

    let merge_requests: Vec<GitlabIssue> = gitlab::api::paged(
        MergeRequests::builder().project(full_name).build()?,
        gitlab::api::Pagination::All,
    )
    .query_async(&gitlab)
    .await?;

    for merge_request in merge_requests {
        let doc = WebDocument {
            source_id: source_id.to_owned(),
            id: merge_request.web_url.clone(),
            link: merge_request.web_url,
            title: merge_request.title,
            body: merge_request.description,
        };
        index.add(doc).await;
    }

    Ok(())
}
