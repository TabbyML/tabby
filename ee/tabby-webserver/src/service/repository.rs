use anyhow::anyhow;
use async_trait::async_trait;
use juniper::ID;
use tabby_common::SourceFile;
use tabby_db::DbConn;

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::schema::{
    repository::{FileEntry, Repository, RepositoryMeta, RepositoryService},
    Result,
};

fn glob_matches(glob: &str, mut input: &str) -> bool {
    for part in glob.split('*') {
        let Some((_, rest)) = input.split_once(part) else {
            return false;
        };
        input = rest;
    }
    true
}

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

    async fn create_repository(&self, name: String, git_url: String) -> Result<ID> {
        Ok((self as &DbConn)
            .create_repository(name, git_url)
            .await?
            .as_id())
    }

    async fn delete_repository(&self, id: &ID) -> Result<bool> {
        Ok((self as &DbConn).delete_repository(id.as_rowid()?).await?)
    }

    async fn update_repository(&self, id: &ID, name: String, git_url: String) -> Result<bool> {
        (self as &DbConn)
            .update_repository(id.as_rowid()?, name, git_url)
            .await?;
        Ok(true)
    }

    async fn search_files(
        &self,
        name: String,
        path_glob: String,
        top_n: usize,
    ) -> Result<Vec<FileEntry>> {
        let git_url = self.get_repository_git_url(name).await?;
        let matching = SourceFile::all()
            .map_err(anyhow::Error::from)?
            .filter_map(|file| {
                if file.git_url == git_url && glob_matches(&path_glob, &file.filepath) {
                    Some(FileEntry {
                        r#type: "file".into(), // Directories are not currently stored in files.jsonl
                        path: file.filepath,
                    })
                } else {
                    None
                }
            })
            .take(top_n)
            .collect();
        Ok(matching)
    }

    async fn repository_meta(&self, name: String, path: String) -> Result<RepositoryMeta> {
        let git_url = self.get_repository_git_url(name).await?;
        SourceFile::all()
            .map_err(anyhow::Error::from)?
            .filter_map(|file| {
                (file.filepath == path && file.git_url == git_url).then(move || file.into())
            })
            .next()
            .ok_or_else(|| anyhow!("File not found").into())
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

    #[tokio::test]
    pub async fn test_repository_mutations() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service: &dyn RepositoryService = &db;

        let id_1 = service
            .create_repository(
                "example".into(),
                "https://github.com/example/example".into(),
            )
            .await
            .unwrap();

        let id_2 = service
            .create_repository(
                "example2".into(),
                "https://github.com/example/example2".into(),
            )
            .await
            .unwrap();

        service
            .create_repository(
                "example3".into(),
                "https://github.com/example/example3".into(),
            )
            .await
            .unwrap();

        assert_eq!(
            service
                .list_repositories(None, None, None, None)
                .await
                .unwrap()
                .len(),
            3
        );

        service.delete_repository(&id_1).await.unwrap();

        assert_eq!(
            service
                .list_repositories(None, None, None, None)
                .await
                .unwrap()
                .len(),
            2
        );

        service
            .update_repository(
                &id_2,
                "Example2".to_string(),
                "https://github.com/example/Example2".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(
            service
                .list_repositories(None, None, None, None)
                .await
                .unwrap()
                .first()
                .unwrap()
                .name,
            "Example2"
        );
    }
}
