use anyhow::Result;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use validator::Validate;

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::schema::repository::{
    CreateRepositoryInput, Repository, RepositoryError, RepositoryService,
};

#[async_trait]
impl RepositoryService for DbConn {
    async fn list_repositories(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Repository>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let repositories = self
            .list_repositories_with_filter(limit, skip_id, backwards)
            .await?;
        Ok(repositories.into_iter().map(Into::into).collect())
    }

    async fn create_repository(
        &self,
        name: String,
        git_url: String,
    ) -> Result<ID, RepositoryError> {
        let input = CreateRepositoryInput { name, git_url };
        input.validate()?;
        Ok(self
            .create_repository(input.name, input.git_url)
            .await?
            .as_id())
    }

    async fn delete_repository(&self, id: &ID) -> Result<bool> {
        self.delete_repository(id.as_rowid()?).await
    }

    async fn update_repository(&self, id: &ID, name: String, git_url: String) -> Result<bool> {
        self.update_repository(id.as_rowid()?, name, git_url)
            .await
            .map(|_| true)
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;

    #[tokio::test]
    pub async fn test_duplicate_repository_error() {
        let db = DbConn::new_in_memory().await.unwrap();

        RepositoryService::create_repository(
            &db,
            "example".into(),
            "https://github.com/example/example".into(),
        )
        .await
        .unwrap();

        let err = RepositoryService::create_repository(
            &db,
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
}
