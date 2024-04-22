use async_trait::async_trait;
use juniper::ID;
use tabby_common::config::RepositoryConfig;
use tabby_db::DbConn;

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::schema::{
    git_repository::{GitRepository, GitRepositoryService},
    repository::FileEntrySearchResult,
    Result,
};

#[async_trait]
impl GitRepositoryService for DbConn {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitRepository>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let repositories = self
            .list_repositories_with_filter(limit, skip_id, backwards)
            .await?;
        Ok(repositories.into_iter().map(Into::into).collect())
    }

    async fn create(&self, name: String, git_url: String) -> Result<ID> {
        Ok(self.create_repository(name, git_url).await?.as_id())
    }

    async fn get_by_name(&self, name: &str) -> Result<GitRepository> {
        Ok(self.get_repository_by_name(name).await?.into())
    }

    async fn delete(&self, id: &ID) -> Result<bool> {
        Ok(self.delete_repository(id.as_rowid()?).await?)
    }

    async fn update(&self, id: &ID, name: String, git_url: String) -> Result<bool> {
        self.update_repository(id.as_rowid()?, name, git_url)
            .await?;
        Ok(true)
    }

    async fn search_files(
        &self,
        name: &str,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>> {
        if pattern.trim().is_empty() {
            return Ok(vec![]);
        }
        let git_url = self.get_repository_by_name(name).await?.git_url;
        let config = RepositoryConfig::new(git_url);

        let pattern = pattern.to_owned();
        let matching = tokio::task::spawn_blocking(move || async move {
            tabby_search::FileSearch::search(&config.dir(), &pattern, top_n)
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
    use temp_testdir::TempDir;

    use super::*;

    #[tokio::test]
    pub async fn test_duplicate_repository_error() {
        let db = DbConn::new_in_memory().await.unwrap();

        GitRepositoryService::create(
            &db,
            "example".into(),
            "https://github.com/example/example".into(),
        )
        .await
        .unwrap();

        let err = GitRepositoryService::create(
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

    #[tokio::test]
    pub async fn test_repository_mutations() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service: &dyn GitRepositoryService = &db;

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

    #[tokio::test]
    pub async fn test_search_files() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service: &dyn GitRepositoryService = &db;

        let dir = TempDir::default();
        let repo_name = "test_repo".to_owned();
        let test_repo_dir = dir.join(&repo_name);
        service
            .create(
                repo_name.clone(),
                format!("file://{}", test_repo_dir.display()),
            )
            .await
            .unwrap();
        tokio::fs::create_dir(&test_repo_dir).await.unwrap();
        tokio::fs::write(test_repo_dir.join("file1.txt"), [])
            .await
            .unwrap();
        tokio::fs::write(test_repo_dir.join("file2.txt"), [])
            .await
            .unwrap();
        tokio::fs::write(test_repo_dir.join("file3.txt"), [])
            .await
            .unwrap();

        let inner = test_repo_dir.join("inner");
        tokio::fs::create_dir(&inner).await.unwrap();
        tokio::fs::write(inner.join("main.rs"), []).await.unwrap();

        let matches: Vec<_> = service
            .search_files(&repo_name, "ex 1", 100)
            .await
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();

        assert!(matches.iter().any(|p| p.contains("file1.txt")));
        assert!(!matches.iter().any(|p| p.contains("file2.txt")));

        let matches: Vec<_> = service
            .search_files(&repo_name, "rs", 10)
            .await
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();

        assert_eq!(matches.len(), 1);
        assert!(matches.iter().any(|p| p.contains("main.rs")));

        let matches: Vec<_> = service
            .search_files(&repo_name, "inner", 10)
            .await
            .unwrap()
            .into_iter()
            .collect();

        assert!(matches.iter().any(|f| f.r#type == "dir"));
        assert_eq!(matches.len(), 2);

        let matches: Vec<_> = service
            .search_files(&repo_name, "", 10)
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(matches.len(), 0);
    }
}
