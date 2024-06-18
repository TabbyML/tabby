mod git;
mod third_party;

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use juniper::ID;
use tabby_common::config::{config_index_to_id, Config, RepositoryConfig};
use tabby_db::DbConn;
use tabby_schema::{
    integration::IntegrationService,
    repository::{
        FileEntrySearchResult, GitRepositoryService, ProvidedRepository, Repository,
        RepositoryKind, RepositoryService, ThirdPartyRepositoryService,
    },
    Result,
};
use tokio::sync::mpsc::UnboundedSender;

use crate::service::background_job::BackgroundJobEvent;

struct RepositoryServiceImpl {
    git: Arc<dyn GitRepositoryService>,
    third_party: Arc<dyn ThirdPartyRepositoryService>,
    config: Vec<RepositoryConfig>,
}

pub fn create(
    db: DbConn,
    integration: Arc<dyn IntegrationService>,
    background: UnboundedSender<BackgroundJobEvent>,
) -> Arc<dyn RepositoryService> {
    Arc::new(RepositoryServiceImpl {
        git: Arc::new(git::create(db.clone(), background.clone())),
        third_party: Arc::new(third_party::create(db, integration, background.clone())),
        config: Config::load()
            .map(|config| config.repositories)
            .unwrap_or_default(),
    })
}

#[async_trait]
impl RepositoryService for RepositoryServiceImpl {
    async fn list_all_repository_urls(&self) -> Result<Vec<RepositoryConfig>> {
        let mut repos: Vec<RepositoryConfig> = self
            .git
            .list(None, None, None, None)
            .await?
            .into_iter()
            .map(|repo| RepositoryConfig::new(repo.git_url))
            .collect();

        repos.extend(
            self.third_party
                .list_repository_configs()
                .await
                .unwrap_or_default(),
        );

        Ok(repos)
    }

    fn git(&self) -> Arc<dyn GitRepositoryService> {
        self.git.clone()
    }

    fn third_party(&self) -> Arc<dyn ThirdPartyRepositoryService> {
        self.third_party.clone()
    }

    async fn repository_list(&self) -> Result<Vec<Repository>> {
        let mut all = vec![];
        all.extend(self.git().repository_list().await?);
        all.extend(self.third_party().repository_list().await?);
        all.extend(
            self.config
                .iter()
                .enumerate()
                .map(|(index, repo)| {
                    Ok(Repository {
                        id: ID::new(config_index_to_id(index)),
                        name: repo.dir_name(),
                        kind: RepositoryKind::Git,
                        dir: repo.dir(),
                        refs: tabby_git::list_refs(&repo.dir())?,
                        git_url: repo.git_url.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        );

        Ok(all)
    }

    async fn resolve_repository(&self, kind: &RepositoryKind, id: &ID) -> Result<Repository> {
        match kind {
            RepositoryKind::Git => self.git().get_repository(id).await,
            RepositoryKind::Github
            | RepositoryKind::Gitlab
            | RepositoryKind::GithubSelfHosted
            | RepositoryKind::GitlabSelfHosted => self
                .third_party()
                .get_provided_repository(id.clone())
                .await
                .map(|repo| to_repository(*kind, repo)),
        }
    }

    async fn search_files(
        &self,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>> {
        if pattern.trim().is_empty() {
            return Ok(vec![]);
        }
        let dir = self.resolve_repository(kind, id).await?.dir;

        let pattern = pattern.to_owned();
        let matching = tabby_git::search_files(&dir, rev, &pattern, top_n)
            .await
            .map(|x| {
                x.into_iter()
                    .map(|f| FileEntrySearchResult {
                        r#type: f.r#type,
                        path: f.path,
                        indices: f.indices,
                    })
                    .collect()
            })
            .map_err(anyhow::Error::from)?;

        Ok(matching)
    }

    async fn grep(
        &self,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        query: &str,
        top_n: usize,
    ) -> Result<Vec<tabby_schema::repository::GrepFile>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        let dir = self.resolve_repository(kind, id).await?.dir;

        let ret = tabby_git::grep(&dir, rev, query)
            .await?
            .map(to_grep_file)
            .take(top_n)
            .collect::<Vec<_>>()
            .await;

        Ok(ret)
    }

    fn configured_repositories(&self) -> Vec<RepositoryConfig> {
        self.config.clone()
    }
}

fn to_grep_file(file: tabby_git::GrepFile) -> tabby_schema::repository::GrepFile {
    tabby_schema::repository::GrepFile {
        path: file.path.display().to_string(),
        lines: file.lines.into_iter().map(to_grep_line).collect(),
    }
}

fn to_grep_line(line: tabby_git::GrepLine) -> tabby_schema::repository::GrepLine {
    tabby_schema::repository::GrepLine {
        line: match line.line {
            tabby_git::GrepTextOrBase64::Text(text) => {
                tabby_schema::repository::GrepTextOrBase64::Text(text)
            }
            tabby_git::GrepTextOrBase64::Base64(bytes) => {
                tabby_schema::repository::GrepTextOrBase64::Base64(bytes)
            }
        },
        byte_offset: line.byte_offset as i32,
        line_number: line.line_number as i32,
        sub_matches: line.sub_matches.into_iter().map(to_sub_match).collect(),
    }
}

fn to_sub_match(m: tabby_git::GrepSubMatch) -> tabby_schema::repository::GrepSubMatch {
    tabby_schema::repository::GrepSubMatch {
        bytes_start: m.bytes_start as i32,
        bytes_end: m.bytes_end as i32,
    }
}

fn list_refs(git_url: &str) -> Vec<String> {
    let dir = RepositoryConfig::new(git_url.to_owned()).dir();
    tabby_git::list_refs(&dir).unwrap_or_default()
}

fn to_repository(kind: RepositoryKind, repo: ProvidedRepository) -> Repository {
    let config = RepositoryConfig::new(&repo.git_url);
    Repository {
        id: repo.id,
        name: repo.display_name,
        kind,
        dir: config.dir(),
        git_url: config.canonical_git_url(),
        refs: list_refs(&repo.git_url),
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;

    fn create_fake() -> UnboundedSender<BackgroundJobEvent> {
        let (sender, _) = tokio::sync::mpsc::unbounded_channel();
        sender
    }

    #[tokio::test]
    async fn test_list_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        let background = create_fake();
        let integration = Arc::new(crate::service::integration::create(db.clone(), background));
        let service = create(db.clone(), integration, create_fake());
        service
            .git()
            .create("test_git_repo".into(), "http://test_git_repo".into())
            .await
            .unwrap();

        // FIXME(boxbeam): add repo with github service once there's syncing logic.
        let repos = service.list_all_repository_urls().await.unwrap();
        assert_eq!(repos.len(), 1);
    }
}
