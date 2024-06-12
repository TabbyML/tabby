use std::sync::Arc;

use anyhow::{anyhow, Result};
use gitlab::{
    api::{issues::ProjectIssues, AsyncQuery},
    GitlabBuilder,
};
use octocrab::Octocrab;
use serde::Deserialize;
use tabby_inference::Embedding;
use tabby_scheduler::{DocIndexer, WebDocument};

pub async fn fetch_github_issues(
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<Vec<WebDocument>> {
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

    Ok(issues
        .into_iter()
        .map(|issue| WebDocument {
            id: format!("github/{full_name}/issues/{}", issue.id),
            link: issue.html_url.to_string(),
            title: issue.title,
            body: issue.body.unwrap_or_default(),
        })
        .collect())
}

#[derive(Deserialize)]
struct GitlabIssue {
    title: String,
    id: u64,
    description: String,
    web_url: String,
}

pub async fn fetch_gitlab_issues(
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<Vec<WebDocument>> {
    let api_base = api_base.strip_prefix("https://").unwrap_or(api_base);
    let gitlab = GitlabBuilder::new(api_base, access_token)
        .build_async()
        .await?;

    let issues: Vec<GitlabIssue> = gitlab::api::paged(
        ProjectIssues::builder().project(full_name).build()?,
        gitlab::api::Pagination::All,
    )
    .query_async(&gitlab)
    .await?;

    Ok(issues
        .into_iter()
        .map(|issue| WebDocument {
            id: format!("gitlab/{full_name}/issues/{}", issue.id),
            link: issue.web_url,
            title: issue.title,
            body: issue.description,
        })
        .collect())
}

pub async fn index_issues(embedding: Arc<dyn Embedding>, issues: Vec<WebDocument>) -> Result<()> {
    let index = DocIndexer::new(embedding);

    for issue in issues {
        index.index_issue(issue).await;
    }

    index.commit();
    Ok(())
}
