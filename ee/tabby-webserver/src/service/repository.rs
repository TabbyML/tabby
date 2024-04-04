use std::path::Path;

use async_trait::async_trait;
use ignore::Walk;
use juniper::ID;
use nucleo::{Utf32Str, Utf32String};
use tabby_common::config::RepositoryConfig;
use tabby_db::DbConn;

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::schema::{
    repository::{FileEntry, Repository, RepositoryService},
    Result,
};

async fn match_glob(base: &Path, glob: &str, limit: usize) -> Result<Vec<FileEntry>, anyhow::Error> {
    let mut nucleo = nucleo::Matcher::new(nucleo::Config::DEFAULT.match_paths());
    let needle = Utf32Str::Ascii(glob.as_bytes());

    let entries: Vec<_> = Walk::new(base)
        .into_iter()
        .filter_map(|path| {
            let entry = path.ok()?;
            let r#type = if entry.file_type().map(|x| x.is_dir()).unwrap_or_default() {
                "dir".into()
            } else {
                "file".into()
            };
            let path = entry
                .into_path()
                .strip_prefix(base)
                .ok()?
                .to_string_lossy()
                .into_owned();
            let haystack: Utf32String = path.clone().into();
            nucleo
                .fuzzy_match(haystack.slice(..), needle)
                .map(|_| FileEntry { r#type, path })
        })
        .take(limit)
        .collect();

    Ok(entries)
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
        Ok(self.create_repository(name, git_url).await?.as_id())
    }

    async fn delete_repository(&self, id: &ID) -> Result<bool> {
        Ok(self.delete_repository(id.as_rowid()? as i64).await?)
    }

    async fn update_repository(&self, id: &ID, name: String, git_url: String) -> Result<bool> {
        self.update_repository(id.as_rowid()? as i64, name, git_url)
            .await?;
        Ok(true)
    }

    async fn search_files(
        &self,
        name: String,
        path_glob: String,
        top_n: usize,
    ) -> Result<Vec<FileEntry>> {
        if path_glob.trim().is_empty() {
            return Ok(vec![]);
        }
        let git_url = self.get_repository_by_name(name.clone()).await?.git_url;
        let config = RepositoryConfig::new_named(name, git_url);
        let matching = match_glob(&config.dir(), &path_glob, top_n)
            .await
            .map_err(anyhow::Error::from)?;
        Ok(matching)
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
