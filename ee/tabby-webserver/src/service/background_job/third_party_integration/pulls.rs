use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;
use octocrab::{models::IssueState, Octocrab};
use tabby_index::public::{
    StructuredDoc, StructuredDocFields, StructuredDocPullDocumentFields, StructuredDocState,
};

use super::error::octocrab_error_message;

// FIXME(kweizh): we can only get StructuredDoc id after constructing the StructuredDoc
// but we need to pass the id to the StructuredDocState
// so we need to refactor the id() method in StructuredDoc
fn pull_id(pull: &octocrab::models::pulls::PullRequest) -> String {
    pull.html_url
        .clone()
        .map(|url| url.to_string())
        .unwrap_or_else(|| pull.url.clone())
}

pub async fn list_github_pull_states(
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<impl Stream<Item = (u64, StructuredDocState)>> {
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
                let id = pull_id(&pull);

                // skip closed but not merged pulls
                if let Some(state) = pull.state {
                    if state == IssueState::Closed && pull.merged_at.is_none() {
                        yield (pull.number, StructuredDocState{
                            id: id,
                            updated_at: pull.updated_at.unwrap(),
                            deleted: true,
                        });
                        continue;
                    }
                }

                yield (pull.number, StructuredDocState{
                    id: id,
                    updated_at: pull.updated_at.unwrap_or_else(|| chrono::Utc::now()),
                    deleted: false,
                });
            }

            page += 1;
            if page > pages {
                break;
            }
        }
    };

    Ok(s)
}

pub async fn get_github_pull_doc(
    source_id: &str,
    id: u64,
    api_base: &str,
    full_name: &str,
    access_token: &str,
) -> Result<StructuredDoc> {
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

    let pull = octocrab.pulls(&owner, &repo).get(id).await.map_err(|e| {
        anyhow!(
            "Failed to fetch pull requests: {}",
            octocrab_error_message(e)
        )
    })?;

    let url = pull
        .html_url
        .map(|url| url.to_string())
        .unwrap_or_else(|| pull.url);
    let title = pull.title.clone().unwrap_or_default();
    let body = pull.body.clone().unwrap_or_default();

    // Fetch the diff only if the number of changed lines is fewer than 100,000,
    // assuming 80 characters per line,
    // and the size of the diff is less than 8MB.
    let diff =
        if pull.additions.unwrap_or_default() + pull.deletions.unwrap_or_default() < 100 * 1024 {
            octocrab
                .pulls(&owner, &repo)
                .get_diff(pull.number)
                .await
                .map_err(|e| {
                    anyhow!(
                        "Failed to fetch pull request diff: {}",
                        octocrab_error_message(e)
                    )
                })?
        } else {
            String::new()
        };

    Ok(StructuredDoc {
        source_id: source_id.to_string(),
        fields: StructuredDocFields::Pull(StructuredDocPullDocumentFields {
            link: url.clone(),
            title,
            body,
            merged: pull.merged_at.is_some(),
            diff,
        }),
    })
}
