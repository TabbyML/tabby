use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    repository::{GitRepository, GitRepositoryService, Repository, RepositoryProvider},
    AsID, AsRowid, Result,
};

use crate::service::{background_job::BackgroundJob, graphql_pagination_to_filter};

struct GitRepositoryServiceImpl {
    db: DbConn,
    background_job: Arc<dyn BackgroundJob>,
}

pub fn create(db: DbConn, background_job: Arc<dyn BackgroundJob>) -> impl GitRepositoryService {
    GitRepositoryServiceImpl { db, background_job }
}

#[async_trait]
impl GitRepositoryService for GitRepositoryServiceImpl {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitRepository>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let repositories = self
            .db
            .list_repositories_with_filter(limit, skip_id, backwards)
            .await?;
        Ok(repositories.into_iter().map(Into::into).collect())
    }

    async fn create(&self, name: String, git_url: String) -> Result<ID> {
        let id = self.db.create_repository(name, git_url).await?.as_id();
        self.background_job.trigger_scheduler().await;
        Ok(id)
    }

    async fn delete(&self, id: &ID) -> Result<bool> {
        Ok(self.db.delete_repository(id.as_rowid()?).await?)
    }

    async fn update(&self, id: &ID, name: String, git_url: String) -> Result<bool> {
        self.db
            .update_repository(id.as_rowid()?, name, git_url)
            .await?;
        self.background_job.trigger_scheduler().await;
        Ok(true)
    }
}

#[async_trait]
impl RepositoryProvider for GitRepositoryServiceImpl {
    async fn repository_list(&self) -> Result<Vec<Repository>> {
        Ok(self
            .list(None, None, None, None)
            .await?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    async fn get_repository(&self, id: &ID) -> Result<Repository> {
        let git_repo: GitRepository = self.db.get_repository(id.as_rowid()?).await?.into();
        Ok(git_repo.into())
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;
    use crate::background_job::create_fake;

    #[tokio::test]
    pub async fn test_duplicate_repository_error() {
        let db = DbConn::new_in_memory().await.unwrap();
        let svc = create(db.clone(), create_fake());

        GitRepositoryService::create(
            &svc,
            "example".into(),
            "https://github.com/example/example".into(),
        )
        .await
        .unwrap();

        let err = GitRepositoryService::create(
            &svc,
            "example".into(),
            "https://github.com/example/example".into(),
        )
        .await
        .unwrap_err();

        assert_eq!(
            err.to_string(),
            "A repository with the same name or URL already exists"
        );
    }

    #[tokio::test]
    pub async fn test_repository_mutations() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone(), create_fake());

        let id_1 = service
            .create(
                "example".into(),
                "https://github.com/example/example".into(),
            )
            .await
            .unwrap();

        let id_2 = service
            .create(
                "example2".into(),
                "https://github.com/example/example2".into(),
            )
            .await
            .unwrap();

        service
            .create(
                "example3".into(),
                "https://github.com/example/example3".into(),
            )
            .await
            .unwrap();

        assert_eq!(service.list(None, None, None, None).await.unwrap().len(), 3);

        service.delete(&id_1).await.unwrap();

        assert_eq!(service.list(None, None, None, None).await.unwrap().len(), 2);

        service
            .update(
                &id_2,
                "Example2".to_string(),
                "https://github.com/example/Example2".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(
            service
                .list(None, None, None, None)
                .await
                .unwrap()
                .first()
                .unwrap()
                .name,
            "Example2"
        );
    }
}
