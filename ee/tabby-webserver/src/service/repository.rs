use std::sync::Arc;

use async_trait::async_trait;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tabby_db::DbConn;

use super::github_repository_provider;
use crate::schema::{
    git_repository::GitRepositoryService,
    github_repository_provider::GithubRepositoryProviderService, repository::RepositoryService,
};

struct RepositoryServiceImpl {
    git: Arc<dyn GitRepositoryService>,
    github: Arc<dyn GithubRepositoryProviderService>,
}

pub fn create(db: DbConn) -> Arc<dyn RepositoryService> {
    Arc::new(RepositoryServiceImpl {
        git: Arc::new(db.clone()),
        github: Arc::new(github_repository_provider::create(db.clone())),
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

        Ok(repos)
    }
}

impl RepositoryService for RepositoryServiceImpl {
    fn access(self: Arc<Self>) -> Arc<dyn RepositoryAccess> {
        self.clone()
    }

    fn git(&self) -> Arc<dyn GitRepositoryService> {
        self.git.clone()
    }

    fn github(&self) -> Arc<dyn GithubRepositoryProviderService> {
        self.github.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tabby_db::DbConn;

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
