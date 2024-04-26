use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tabby_db::DbConn;

use super::{github_repository_provider, gitlab_repository_provider, Result};
use crate::schema::{
    git_repository::GitRepositoryService,
    github_repository_provider::GithubRepositoryProviderService,
    gitlab_repository::GitlabRepositoryService,
    repository::{FileEntrySearchResult, Repository, RepositoryKind, RepositoryService},
};

struct RepositoryServiceImpl {
    git: Arc<dyn GitRepositoryService>,
    github: Arc<dyn GithubRepositoryProviderService>,
    gitlab: Arc<dyn GitlabRepositoryService>,
}

pub fn create(db: DbConn) -> Arc<dyn RepositoryService> {
    Arc::new(RepositoryServiceImpl {
        git: Arc::new(db.clone()),
        github: Arc::new(github_repository_provider::create(db.clone())),
        gitlab: Arc::new(gitlab_repository_provider::create(db.clone())),
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
            self.github
                .list_provided_git_urls()
                .await
                .unwrap_or_default()
                .into_iter()
                .map(RepositoryConfig::new),
        );

        repos.extend(
            self.gitlab
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

    fn github(&self) -> Arc<dyn GithubRepositoryProviderService> {
        self.github.clone()
    }

    fn gitlab(&self) -> Arc<dyn GitlabRepositoryService> {
        self.gitlab.clone()
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
            tabby_search::FileSearch::search(&dir, &pattern, top_n)
                .map(|x| x.into_iter().map(|f| f.into()).collect())
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

    use super::*;

    #[tokio::test]
    async fn test_list_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());
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
