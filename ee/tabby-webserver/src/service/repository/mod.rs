mod git;
mod third_party;

use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_schema::{
    integration::IntegrationService,
    repository::{
        FileEntrySearchResult, GitRepositoryService, Repository, RepositoryKind, RepositoryService,
        ThirdPartyRepositoryService,
    },
    Result,
};
use tokio::sync::mpsc::UnboundedSender;

use crate::service::background_job::BackgroundJobEvent;

struct RepositoryServiceImpl {
    git: Arc<dyn GitRepositoryService>,
    third_party: Arc<dyn ThirdPartyRepositoryService>,
}

pub fn create(
    db: DbConn,
    integration: Arc<dyn IntegrationService>,
    background: UnboundedSender<BackgroundJobEvent>,
) -> Arc<dyn RepositoryService> {
    Arc::new(RepositoryServiceImpl {
        git: Arc::new(git::create(db.clone(), background.clone())),
        third_party: Arc::new(third_party::create(db, integration, background.clone())),
    })
}

#[async_trait]
impl RepositoryAccess for RepositoryServiceImpl {
    async fn list_repositories(&self) -> anyhow::Result<Vec<RepositoryConfig>> {
        let mut repos: Vec<RepositoryConfig> = self
            .git
            .list(None, None, None, None)
            .await?
            .into_iter()
            .map(|repo| RepositoryConfig::new(repo.git_url))
            .collect();

        repos.extend(
            self.third_party
                .list_active_git_urls()
                .await
                .unwrap_or_default()
                .into_iter()
                .map(RepositoryConfig::new),
        );

        Ok(repos)
    }
}

#[async_trait]
impl RepositoryService for RepositoryServiceImpl {
    fn git(&self) -> Arc<dyn GitRepositoryService> {
        self.git.clone()
    }

    fn third_party(&self) -> Arc<dyn ThirdPartyRepositoryService> {
        self.third_party.clone()
    }

    fn access(self: Arc<Self>) -> Arc<dyn RepositoryAccess> {
        self.clone()
    }

    async fn repository_list(&self) -> Result<Vec<Repository>> {
        let mut all = vec![];
        all.extend(self.git().repository_list().await?);
        all.extend(self.third_party().repository_list().await?);

        Ok(all)
    }

    async fn resolve_repository(&self, kind: &RepositoryKind, id: &ID) -> Result<Repository> {
        match kind {
            RepositoryKind::Git => self.git().get_repository(id).await,
            RepositoryKind::Github => {
                self.third_party()
                    .get_repository(id.clone())
                    .await
                    .map(|repo| Repository {
                        id: repo.id,
                        name: repo.display_name,
                        kind: RepositoryKind::Github,
                        refs: list_refs(&repo.git_url),
                        dir: RepositoryConfig::new(repo.git_url).dir(),
                    })
            }
            RepositoryKind::Gitlab => {
                self.third_party()
                    .get_repository(id.clone())
                    .await
                    .map(|repo| Repository {
                        id: repo.id,
                        name: repo.display_name,
                        kind: RepositoryKind::Gitlab,
                        refs: list_refs(&repo.git_url),
                        dir: RepositoryConfig::new(repo.git_url).dir(),
                    })
            }
        }
    }

    async fn search_files(
        &self,
        kind: &RepositoryKind,
        id: &ID,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>> {
        if pattern.trim().is_empty() {
            return Ok(vec![]);
        }
        let dir = self.resolve_repository(kind, id).await?.dir;

        let pattern = pattern.to_owned();
        let matching = tabby_git::search_files(&dir, None, &pattern, top_n)
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
}

fn list_refs(git_url: &str) -> Vec<String> {
    let dir = RepositoryConfig::new(git_url.to_owned()).dir();
    tabby_git::list_refs(&dir).unwrap_or_default()
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
        let repos = service.list_repositories().await.unwrap();
        assert_eq!(repos.len(), 1);
    }
}
