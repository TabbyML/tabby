use octocrab::Octocrab;

use super::RepositoryInfo;

pub async fn fetch_all_github_repos(
    access_token: &str,
    base_url: &str,
) -> Result<Vec<RepositoryInfo>, octocrab::Error> {
    let octocrab = Octocrab::builder()
        .base_uri(base_url)?
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
