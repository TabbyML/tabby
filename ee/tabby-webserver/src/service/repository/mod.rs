mod git;
mod prompt_tools;
mod third_party;

use std::sync::Arc;

use async_trait::async_trait;
use cached::{CachedAsync, TimedCache};
use futures::StreamExt;
use juniper::ID;
use prompt_tools::pipeline_related_questions_with_repo_dirs;
use tabby_common::config::{
    config_id_to_index, config_index_to_id, CodeRepository, Config, RepositoryConfig,
};
use tabby_db::DbConn;
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    bail,
    integration::IntegrationService,
    job::JobService,
    policy::AccessPolicy,
    repository::{
        FileEntrySearchResult, GitReference, GitRepositoryService, ProvidedRepository, Repository,
        RepositoryKind, RepositoryService, ThirdPartyRepositoryService,
    },
    Result,
};
use tokio::sync::Mutex;

struct RepositoryServiceImpl {
    git: Arc<dyn GitRepositoryService>,
    third_party: Arc<dyn ThirdPartyRepositoryService>,
    config: Vec<RepositoryConfig>,
    related_questions_cache: Mutex<TimedCache<String, Vec<String>>>,
}

const RELATED_QUESTIONS_CACHE_LIFESPAN: u64 = 60 * 30; // 30 minutes

pub fn create(
    db: DbConn,
    integration: Arc<dyn IntegrationService>,
    job: Arc<dyn JobService>,
) -> Arc<dyn RepositoryService> {
    Arc::new(RepositoryServiceImpl {
        git: Arc::new(git::create(db.clone(), job.clone())),
        third_party: Arc::new(third_party::create(db, integration, job.clone())),
        config: Config::load()
            .map(|config| config.repositories)
            .unwrap_or_default(),
        related_questions_cache: Mutex::new(TimedCache::with_lifespan(
            RELATED_QUESTIONS_CACHE_LIFESPAN,
        )),
    })
}

impl RepositoryServiceImpl {
    async fn find_repository_by_source_id(
        &self,
        policy: &AccessPolicy,
        source_id: &str,
    ) -> Result<Repository> {
        let repositories = self.repository_list(Some(policy)).await?;
        for repository in repositories {
            if repository.source_id == source_id {
                return Ok(repository);
            }
        }
        bail!(
            "Repository not found or 'source_id' is invalid: {}",
            source_id
        )
    }
}

#[async_trait]
impl RepositoryService for RepositoryServiceImpl {
    async fn read_repository_related_questions(
        &self,
        chat: Arc<dyn ChatCompletionStream>,
        policy: &AccessPolicy,
        source_id: String,
    ) -> Result<Vec<String>> {
        if source_id.is_empty() {
            return Err(anyhow::anyhow!("Invalid source_id format"))?;
        }

        let mut cache = self.related_questions_cache.lock().await;
        let questions = cache
            .try_get_or_set_with(source_id.clone(), || async {
                let repository = self
                    .find_repository_by_source_id(policy, &source_id)
                    .await?;

                let (files, truncated) = match self
                    .list_files(policy, &repository.kind, &repository.id, None, Some(300))
                    .await
                {
                    Ok((files, truncated)) => (files, truncated),
                    Err(_) => {
                        return Err(anyhow::anyhow!(
                            "Repository exists but not accessible: {}",
                            source_id
                        ))?
                    }
                };

                pipeline_related_questions_with_repo_dirs(chat, &repository, files, truncated).await
            })
            .await?;
        Ok(questions.to_owned())
    }

    async fn list_all_code_repository(&self) -> Result<Vec<CodeRepository>> {
        // Read repositories configured as git url.
        let mut repos: Vec<CodeRepository> = self
            .git
            .list(None, None, None, None)
            .await?
            .into_iter()
            .map(|repo| CodeRepository::new(&repo.git_url, &repo.source_id()))
            .collect();

        // Read repositories configured as third party integration (e.g Github, Gitlab)
        repos.extend(
            self.third_party
                .list_code_repositories()
                .await
                .unwrap_or_default(),
        );

        // Read repositories configured in `config.toml`
        repos.extend(
            self.config
                .iter()
                .enumerate()
                .map(|(index, repo)| {
                    CodeRepository::new(repo.git_url(), &config_index_to_id(index))
                })
                .collect::<Vec<CodeRepository>>(),
        );

        Ok(repos)
    }

    fn git(&self) -> Arc<dyn GitRepositoryService> {
        self.git.clone()
    }

    fn third_party(&self) -> Arc<dyn ThirdPartyRepositoryService> {
        self.third_party.clone()
    }

    async fn repository_list(&self, policy: Option<&AccessPolicy>) -> Result<Vec<Repository>> {
        let mut all = vec![];
        all.extend(self.git().repository_list().await?);
        all.extend(self.third_party().repository_list().await?);

        all.extend(
            self.config
                .iter()
                .enumerate()
                .map(|(index, repo)| repository_config_to_repository(index, repo))
                .collect::<Vec<_>>(),
        );

        if let Some(policy) = policy {
            // Only keep repositories that the user has read access to.
            let mut filtered = Vec::new();
            for repo in all {
                if policy.check_read_source(&repo.source_id).await.is_ok() {
                    filtered.push(repo);
                }
            }
            all = filtered;
        }

        Ok(all)
    }

    async fn resolve_repository(
        &self,
        policy: &AccessPolicy,
        kind: &RepositoryKind,
        id: &ID,
    ) -> Result<Repository> {
        let ret = match kind {
            RepositoryKind::GitConfig => {
                let index = config_id_to_index(id)?;
                let config = &self.config[index];
                return Ok(repository_config_to_repository(index, config));
            }
            RepositoryKind::Git => self.git().get_repository(id).await,
            RepositoryKind::Github
            | RepositoryKind::Gitlab
            | RepositoryKind::GithubSelfHosted
            | RepositoryKind::GitlabSelfHosted => self
                .third_party()
                .get_provided_repository(id)
                .await
                .map(|repo| to_repository(*kind, repo)),
        };

        match ret {
            Ok(repo) => {
                policy.check_read_source(&repo.source_id).await?;
                Ok(repo)
            }
            _ => ret,
        }
    }

    async fn search_files(
        &self,
        policy: &AccessPolicy,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>> {
        if pattern.trim().is_empty() {
            return Ok(vec![]);
        }
        let dir = self.resolve_repository(policy, kind, id).await?.dir;

        let pattern = pattern.to_owned();
        let matching = tabby_git::search_files(&dir, rev, &pattern, top_n)
            .await
            .map(|x| {
                x.into_iter()
                    .map(|f| FileEntrySearchResult {
                        r#type: f.r#type,
                        path: f.path,
                        indices: f.indices,
                    })
                    .collect()
            })
            .map_err(anyhow::Error::from)?;

        Ok(matching)
    }

    async fn list_files(
        &self,
        policy: &AccessPolicy,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        top_n: Option<usize>,
    ) -> Result<(Vec<FileEntrySearchResult>, bool)> {
        let dir = self.resolve_repository(policy, kind, id).await?.dir;
        let (files, truncated) = tabby_git::list_files(&dir, rev, top_n)
            .await
            .map(|list_file| {
                let files = list_file
                    .files
                    .into_iter()
                    .map(|f| FileEntrySearchResult {
                        r#type: f.r#type,
                        path: f.path,
                        indices: f.indices,
                    })
                    .collect();
                (files, list_file.truncated)
            })
            .map_err(anyhow::Error::from)?;
        Ok((files, truncated))
    }

    async fn grep(
        &self,
        policy: &AccessPolicy,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        query: &str,
        top_n: usize,
    ) -> Result<Vec<tabby_schema::repository::GrepFile>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        let dir = self.resolve_repository(policy, kind, id).await?.dir;

        let ret = tabby_git::grep(&dir, rev, query)
            .await?
            .map(to_grep_file)
            .take(top_n)
            .collect::<Vec<_>>()
            .await;

        Ok(ret)
    }
}

fn to_grep_file(file: tabby_git::GrepFile) -> tabby_schema::repository::GrepFile {
    tabby_schema::repository::GrepFile {
        path: file.path.display().to_string(),
        lines: file.lines.into_iter().map(to_grep_line).collect(),
    }
}

fn to_grep_line(line: tabby_git::GrepLine) -> tabby_schema::repository::GrepLine {
    tabby_schema::repository::GrepLine {
        line: match line.line {
            tabby_git::GrepTextOrBase64::Text(text) => {
                tabby_schema::repository::GrepTextOrBase64::Text(text)
            }
            tabby_git::GrepTextOrBase64::Base64(bytes) => {
                tabby_schema::repository::GrepTextOrBase64::Base64(bytes)
            }
        },
        byte_offset: line.byte_offset as i32,
        line_number: line.line_number as i32,
        sub_matches: line.sub_matches.into_iter().map(to_sub_match).collect(),
    }
}

fn to_sub_match(m: tabby_git::GrepSubMatch) -> tabby_schema::repository::GrepSubMatch {
    tabby_schema::repository::GrepSubMatch {
        bytes_start: m.bytes_start as i32,
        bytes_end: m.bytes_end as i32,
    }
}

fn list_refs(git_url: &str) -> Vec<tabby_git::GitReference> {
    let dir = RepositoryConfig::resolve_dir(git_url);
    tabby_git::list_refs(&dir).unwrap_or_default()
}

fn to_repository(kind: RepositoryKind, repo: ProvidedRepository) -> Repository {
    Repository {
        source_id: repo.source_id(),
        id: ID::new(repo.source_id()),
        name: repo.display_name,
        kind,
        dir: RepositoryConfig::resolve_dir(&repo.git_url),
        git_url: RepositoryConfig::canonicalize_url(&repo.git_url),
        refs: list_refs(&repo.git_url)
            .into_iter()
            .map(|r| GitReference {
                name: r.name,
                commit: r.commit,
            })
            .collect(),
    }
}

fn repository_config_to_repository(index: usize, config: &RepositoryConfig) -> Repository {
    let source_id = config_index_to_id(index);
    Repository {
        id: ID::new(source_id.clone()),
        source_id,
        name: config.display_name(),
        kind: RepositoryKind::GitConfig,
        dir: config.dir(),
        refs: tabby_git::list_refs(&config.dir())
            .unwrap_or_default()
            .into_iter()
            .map(|r| GitReference {
                name: r.name,
                commit: r.commit,
            })
            .collect(),
        git_url: config.git_url().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;
    use crate::job;

    #[tokio::test]
    async fn test_list_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        let job_service = Arc::new(job::create(db.clone()).await);
        let integration = Arc::new(crate::service::integration::create(db.clone(), job_service));
        let job = Arc::new(crate::service::job::create(db.clone()).await);
        let service = create(db.clone(), integration, job);
        service
            .git()
            .create("test_git_repo".into(), "http://test_git_repo".into())
            .await
            .unwrap();

        // FIXME(boxbeam): add repo with github service once there's syncing logic.
        let repos = service.list_all_code_repository().await.unwrap();
        assert_eq!(repos.len(), 1);
    }
}
