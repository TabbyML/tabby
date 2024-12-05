use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_common::config::{CodeRepository, RepositoryConfig};
use tabby_db::{DbConn, RepositoryDAO};
use tabby_schema::{
    job::{JobInfo, JobService},
    repository::{
        GitReference, GitRepository, GitRepositoryService, Repository, RepositoryProvider,
    },
    AsID, AsRowid, Result,
};

use crate::service::{background_job::BackgroundJobEvent, graphql_pagination_to_filter};

struct GitRepositoryServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
}

pub fn create(db: DbConn, job_service: Arc<dyn JobService>) -> impl GitRepositoryService {
    GitRepositoryServiceImpl { db, job_service }
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

        let mut converted_repositories = vec![];

        for repository in repositories {
            let event = BackgroundJobEvent::SchedulerGitRepository(CodeRepository::new(
                &repository.git_url,
                &GitRepository::format_source_id(&repository.id.as_id()),
            ));
            let job_info = self.job_service.get_job_info(event.to_command()).await?;

            converted_repositories.push(to_git_repository(repository, job_info));
        }
        Ok(converted_repositories)
    }

    async fn create(&self, name: String, git_url: String) -> Result<ID> {
        let id = self
            .db
            .create_repository(name, git_url.clone())
            .await?
            .as_id();
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::SchedulerGitRepository(CodeRepository::new(
                    &git_url,
                    &GitRepository::format_source_id(&id),
                ))
                .to_command(),
            )
            .await;
        Ok(id)
    }

    async fn delete(&self, id: &ID) -> Result<bool> {
        let rowid = id.as_rowid()?;
        let repository = self.db.get_repository(rowid).await?;
        let success = self.db.delete_repository(rowid).await?;
        if success {
            self.job_service
                .clear(
                    BackgroundJobEvent::SchedulerGitRepository(CodeRepository::new(
                        &repository.git_url,
                        &GitRepository::format_source_id(id),
                    ))
                    .to_command(),
                )
                .await?;
            self.job_service
                .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
                .await?;
        }
        Ok(success)
    }

    async fn update(&self, id: &ID, name: String, git_url: String) -> Result<bool> {
        self.db
            .update_repository(id.as_rowid()?, name, git_url.clone())
            .await?;
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::SchedulerGitRepository(CodeRepository::new(
                    &git_url,
                    &GitRepository::format_source_id(id),
                ))
                .to_command(),
            )
            .await;
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
        let dao = self.db.get_repository(id.as_rowid()?).await?;

        let event = BackgroundJobEvent::SchedulerGitRepository(CodeRepository::new(
            &dao.git_url,
            &GitRepository::format_source_id(&dao.id.as_id()),
        ));

        let job_info = self.job_service.get_job_info(event.to_command()).await?;
        let git_repo = to_git_repository(dao, job_info);
        Ok(git_repo.into())
    }
}

fn to_git_repository(repo: RepositoryDAO, job_info: JobInfo) -> GitRepository {
    GitRepository {
        id: repo.id.as_id(),
        name: repo.name,
        refs: tabby_git::list_refs(&RepositoryConfig::resolve_dir(&repo.git_url))
            .unwrap_or_default()
            .into_iter()
            .map(|r| GitReference {
                name: r.name,
                commit: r.commit,
            })
            .collect(),
        git_url: repo.git_url,
        job_info,
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;

    #[tokio::test]
    pub async fn test_duplicate_repository_error() {
        let db = DbConn::new_in_memory().await.unwrap();
        let svc = create(
            db.clone(),
            Arc::new(crate::service::job::create(db.clone()).await),
        );

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
        let job = Arc::new(crate::service::job::create(db.clone()).await);
        let service = create(db.clone(), job);

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
