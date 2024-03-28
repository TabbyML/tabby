use std::path::Path;

use anyhow::anyhow;
use async_trait::async_trait;
use futures::StreamExt;
use juniper::ID;
use tabby_common::{config::RepositoryConfig, SourceFile};
use tabby_db::DbConn;

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::schema::{
    repository::{FileEntry, Repository, RepositoryMeta, RepositoryService},
    Result,
};

fn match_glob(path: &str, glob: &str) -> bool {
    let glob = glob.to_lowercase();
    let mut path = &*path.to_lowercase();
    // Current behavior: Find each "word" (any substring separated by whitespace) appearing in the same order
    // they were entered within the path. If one cannot be found or they are out-of-order, it will not match.
    // To change this behavior, change `.split_whitespace()` to `.chars()` to make it match the characters in
    // order, or `.replace(' ', "").chars()` to do the same while ignoring spaces
    for part in glob.split_whitespace() {
        let Some((_, rest)) = path.split_once(part) else {
            return false;
        };
        path = rest;
    }
    true
}

async fn find_glob(base: &Path, glob: &str, limit: usize) -> Result<Vec<FileEntry>, anyhow::Error> {
    let mut paths = vec![];
    let mut walk = async_walkdir::WalkDir::new(base);
    while let Some(path) = walk.next().await.transpose()? {
        let full_path = path.path();
        let full_path = full_path.strip_prefix(base)?;
        let name = full_path.to_string_lossy();
        if !match_glob(&name, glob) {
            continue;
        }
        paths.push(FileEntry {
            r#type: if path.file_type().await?.is_dir() {
                "dir".into()
            } else {
                "file".into()
            },
            path: name.to_string(),
        });
        if paths.len() >= limit {
            break;
        }
    }
    Ok(paths)
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
        let git_url = self.get_repository_git_url(name.clone()).await?;
        let config = RepositoryConfig::new_named(name, git_url);
        let matching = find_glob(&config.dir(), &path_glob, top_n)
            .await
            .map_err(anyhow::Error::from)?;
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
