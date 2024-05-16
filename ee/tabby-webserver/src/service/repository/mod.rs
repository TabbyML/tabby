mod git;
mod github;
mod gitlab;
mod third_party;

use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_schema::{
    integration::IntegrationService,
    repository::{
        FileEntrySearchResult, GitRepositoryService, GithubRepositoryService,
        GitlabRepositoryService, Repository, RepositoryKind, RepositoryService,
        ThirdPartyRepositoryService,
    },
    Result,
};
use tokio::sync::mpsc::UnboundedSender;

use crate::service::background_job::BackgroundJobEvent;

struct RepositoryServiceImpl {
    git: Arc<dyn GitRepositoryService>,
    third_party: Arc<dyn ThirdPartyRepositoryService>,
    github: Arc<dyn GithubRepositoryService>,
    gitlab: Arc<dyn GitlabRepositoryService>,
}

pub fn create(
    db: DbConn,
    integration: Arc<dyn IntegrationService>,
    background: UnboundedSender<BackgroundJobEvent>,
) -> Arc<dyn RepositoryService> {
    Arc::new(RepositoryServiceImpl {
        git: Arc::new(git::create(db.clone(), background.clone())),
        github: Arc::new(github::create(db.clone(), background.clone())),
        gitlab: Arc::new(gitlab::create(db.clone(), background.clone())),
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

    fn github(&self) -> Arc<dyn GithubRepositoryService> {
        self.github.clone()
    }

    fn gitlab(&self) -> Arc<dyn GitlabRepositoryService> {
        self.gitlab.clone()
    }

    fn third_party(&self) -> Arc<dyn ThirdPartyRepositoryService> {
        self.third_party.clone()
    }

    fn access(self: Arc<Self>) -> Arc<dyn RepositoryAccess> {
        self.clone()
    }

    async fn repository_list(&self) -> Result<Vec<Repository>> {
        let mut all = vec![];
        all.append(&mut self.git().repository_list().await?);
        all.append(&mut self.github().repository_list().await?);
        all.append(&mut self.gitlab().repository_list().await?);

        Ok(all)
    }

    async fn resolve_repository(&self, kind: &RepositoryKind, id: &ID) -> Result<Repository> {
        match kind {
            RepositoryKind::Git => self.git().get_repository(id).await,
            RepositoryKind::Github => self.github().get_repository(id).await,
            RepositoryKind::Gitlab => self.gitlab().get_repository(id).await,
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
        let matching = tokio::task::spawn_blocking(move || async move {
            tabby_search::FileSearch::search(&dir, &pattern, top_n).map(|x| {
                x.into_iter()
                    .map(|f| FileEntrySearchResult {
                        r#type: f.r#type,
                        path: f.path,
                        indices: f.indices,
                    })
                    .collect()
            })
        })
        .await
        .map_err(anyhow::Error::from)?
        .await?;

        Ok(matching)
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;
    use tokio::sync::mpsc::WeakUnboundedSender;
    use tracing::instrument::WithSubscriber;

    use super::*;

    fn create_fake() -> UnboundedSender<BackgroundJobEvent> {
        let (sender, _) = tokio::sync::mpsc::unbounded_channel();
        sender
    }

    #[tokio::test]
    async fn test_list_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        let (background, _) = tokio::sync::mpsc::unbounded_channel();
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
