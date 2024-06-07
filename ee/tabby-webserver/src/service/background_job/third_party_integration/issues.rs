use std::sync::Arc;

use anyhow::{anyhow, Result};
use octocrab::{models::issues::Issue, Octocrab};
use tabby_inference::Embedding;
use tabby_scheduler::{DocIndexer, SourceDocument};

async fn fetch_github_issues(
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<Vec<Issue>> {
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

    Ok(issues)
}

pub async fn index_github_issues(
    embedding: Arc<dyn Embedding>,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<()> {
    let index = DocIndexer::new(embedding);

    for issue in fetch_github_issues(api_base, full_name, access_token).await? {
        let document = SourceDocument {
            id: format!("{full_name}/issues/{}", issue.id),
            title: issue.title,
            link: issue.url.to_string(),
            body: issue.body.unwrap_or_default(),
        };
        index.index_document(document).await;
    }
    Ok(())
}
