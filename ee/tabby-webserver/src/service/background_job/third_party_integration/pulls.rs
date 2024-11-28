use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;
use octocrab::{models::IssueState, Octocrab};
use tabby_index::public::{
    StructuredDoc, StructuredDocFields, StructuredDocPullDocumentFields, StructuredDocState,
};

use super::error::octocrab_error_message;

pub async fn list_github_pulls(
    source_id: &str,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<impl Stream<Item = (StructuredDocState, StructuredDoc)>> {
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
                let url = pull.html_url.map(|url| url.to_string()).unwrap_or_else(|| pull.url);
                let title = pull.title.clone().unwrap_or_default();
                let body = pull.body.clone().unwrap_or_default();
                let doc = StructuredDoc {
                    source_id: source_id.to_string(),
                    fields: StructuredDocFields::Pull(StructuredDocPullDocumentFields {
                        link: url.clone(),
                        title,
                        body,
                        merged: pull.merged_at.is_some(),
                        diff: String::new(),
                    }),
                };

                // skip closed but not merged pulls
                if let Some(state) = pull.state {
                    if state == IssueState::Closed && pull.merged_at.is_none() {
                        yield (StructuredDocState{
                            updated_at: pull.updated_at.unwrap(),
                            deleted: true,
                        }, doc);
                        continue;
                    }
                }

                // Fetch the diff only if the number of changed lines is fewer than 100,000,
                // assuming 80 characters per line,
                // and the size of the diff is less than 8MB.
                let diff = if pull.additions.unwrap_or_default() + pull.deletions.unwrap_or_default()  < 100*1024 {
                    match octocrab.pulls(&owner, &repo).get_diff(pull.number).await {
                        Ok(x) => x,
                        Err(e) => {
                            logkit::error!("Failed to fetch pull request diff for {}: {}",
                                url, octocrab_error_message(e));
                            continue
                        }
                    }
                } else {
                    String::new()
                };

                let doc = StructuredDoc {
                    source_id: source_id.to_string(),
                    fields: StructuredDocFields::Pull(StructuredDocPullDocumentFields {
                        link: url,
                        title: pull.title.unwrap_or_default(),
                        body: pull.body.unwrap_or_default(),
                        diff,
                        merged: pull.merged_at.is_some(),
                })};


                yield (StructuredDocState{
                    updated_at: pull.updated_at.unwrap(),
                    deleted: false,
                }, doc);
            }

            page += 1;
            if page > pages {
                break;
            }
        }
    };

    Ok(s)
}
