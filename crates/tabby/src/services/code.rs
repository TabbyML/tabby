use std::{sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use cached::{CachedAsync, TimedCache};
use parse_git_url::GitUrl;
use tabby_common::{
    api::code::{CodeSearch, CodeSearchError, Hit, HitDocument, SearchResponse},
    config::{RepositoryAccess, RepositoryConfig},
    index::{self, register_tokenizers, CodeSearchSchema},
    path,
};
use tantivy::{
    collector::{Count, TopDocs},
    query::{BooleanQuery, QueryParser},
    query_grammar::Occur,
    schema::Field,
    DocAddress, Document, Index, IndexReader,
};
use textdistance::str::damerau_levenshtein;
use tokio::{sync::Mutex, time::sleep};
use tracing::{debug, log::info};

struct CodeSearchImpl {
    reader: IndexReader,
    query_parser: QueryParser,

    schema: CodeSearchSchema,
    repository_access: Arc<dyn RepositoryAccess>,
    repo_cache: Mutex<TimedCache<(), Vec<RepositoryConfig>>>,
}

impl CodeSearchImpl {
    fn load(repository_access: Arc<dyn RepositoryAccess>) -> Result<Self> {
        let code_schema = index::CodeSearchSchema::new();
        let index = Index::open_in_dir(path::index_dir())?;
        register_tokenizers(&index);

        let query_parser = QueryParser::new(
            code_schema.schema.clone(),
            vec![code_schema.field_body],
            index.tokenizers().clone(),
        );
        let reader = index
            .reader_builder()
            .reload_policy(tantivy::ReloadPolicy::OnCommit)
            .try_into()?;
        Ok(Self {
            repository_access,
            reader,
            query_parser,
            schema: code_schema,
            repo_cache: Mutex::new(TimedCache::with_lifespan(10 * 60)),
        })
    }

    async fn load_async(repository_access: Arc<dyn RepositoryAccess>) -> CodeSearchImpl {
        loop {
            match CodeSearchImpl::load(repository_access.clone()) {
                Ok(code) => {
                    info!("Index is ready, enabling server...");
                    return code;
                }
                Err(err) => {
                    debug!("Source code index is not ready `{}`", err);
                }
            };

            sleep(Duration::from_secs(60)).await;
        }
    }

    fn create_hit(&self, score: f32, doc: Document, doc_address: DocAddress) -> Hit {
        Hit {
            score,
            doc: HitDocument {
                body: get_field(&doc, self.schema.field_body),
                filepath: get_field(&doc, self.schema.field_filepath),
                git_url: get_field(&doc, self.schema.field_git_url),
                language: get_field(&doc, self.schema.field_language),
            },
            id: doc_address.doc_id,
        }
    }

    async fn search_with_query(
        &self,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        let searcher = self.reader.searcher();
        let (top_docs, num_hits) =
            { searcher.search(q, &(TopDocs::with_limit(limit).and_offset(offset), Count))? };
        let hits: Vec<Hit> = {
            top_docs
                .iter()
                .map(|(score, doc_address)| {
                    let doc = searcher.doc(*doc_address).unwrap();
                    self.create_hit(*score, doc, *doc_address)
                })
                .collect()
        };
        Ok(SearchResponse { num_hits, hits })
    }
}

#[async_trait]
impl CodeSearch for CodeSearchImpl {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        let query = self.query_parser.parse_query(q)?;
        self.search_with_query(&query, limit, offset).await
    }

    async fn search_in_language(
        &self,
        git_url: &str,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        let language_query = self.schema.language_query(language);
        let body_query = self.schema.body_query(tokens);

        let mut cache = self.repo_cache.lock().await;

        let repos = cache
            .try_get_or_set_with((), || async {
                let repos = self.repository_access.list_repositories().await?;
                Ok::<_, anyhow::Error>(repos)
            })
            .await?;

        let Some(git_url) = closest_match(git_url, repos.iter()) else {
            return Ok(SearchResponse::default());
        };

        let git_url_query = self.schema.git_url_query(&git_url);

        let query = BooleanQuery::new(vec![
            (Occur::Must, language_query),
            (Occur::Must, body_query),
            (Occur::Must, git_url_query),
        ]);
        self.search_with_query(&query, limit, offset).await
    }
}

fn closest_match<'a>(
    search_term: &'a str,
    search_input: impl IntoIterator<Item = &'a RepositoryConfig>,
) -> Option<String> {
    let git_search = GitUrl::parse(search_term).ok()?;

    let candidates: Vec<_> = search_input
        .into_iter()
        // Name is scored the highest, an exact match is required.
        .filter(|elem| GitUrl::parse(&elem.git_url).is_ok_and(|url| url.name == git_search.name))
        .collect();

    // If we only have a single repository with an exact name match, return that with no further checks
    if let [candidate] = &*candidates {
        return Some(candidate.git_url.clone());
    }

    candidates
        .into_iter()
        .filter_map(|elem| {
            let git_url = GitUrl::parse(&elem.git_url).ok()?;
            let mut score = 0;
            // Host is scored the second highest
            if let Some((host1, host2)) = git_search.host.as_ref().zip(git_url.host) {
                score += 8 * damerau_levenshtein(host1, &host2);
            } else {
                score += 30;
            }

            // Owner is scored the lowest / least important
            if let Some((org1, org2)) = git_search.owner.as_ref().zip(git_url.owner) {
                score += 6 * damerau_levenshtein(org1, &org2);
            } else {
                score += 15;
            }

            dbg!(score);
            Some((score, elem))
        })
        // Higher scores mean a higher degree of difference in the repositories,
        // so we want to eliminate scores over 50 and take the lowest
        .filter(|(score, _elem)| *score < 50)
        .min_by_key(|(score, _elem)| *score)
        .map(|(_score, elem)| elem.git_url.clone())
}

fn get_field(doc: &Document, field: Field) -> String {
    doc.get_first(field)
        .and_then(|x| x.as_text())
        .expect("Missing field")
        .to_owned()
}

struct CodeSearchService {
    search: Arc<Mutex<Option<CodeSearchImpl>>>,
}

impl CodeSearchService {
    pub fn new(repository_access: Arc<dyn RepositoryAccess>) -> Self {
        let search = Arc::new(Mutex::new(None));

        let ret = Self {
            search: search.clone(),
        };

        tokio::spawn(async move {
            let code = CodeSearchImpl::load_async(repository_access).await;
            *search.lock().await = Some(code);
        });

        ret
    }
}

pub fn create_code_search(repository_access: Arc<dyn RepositoryAccess>) -> impl CodeSearch {
    CodeSearchService::new(repository_access)
}

#[async_trait]
impl CodeSearch for CodeSearchService {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        if let Some(imp) = self.search.lock().await.as_ref() {
            imp.search(q, limit, offset).await
        } else {
            Err(CodeSearchError::NotReady)
        }
    }

    async fn search_in_language(
        &self,
        git_url: &str,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        if let Some(imp) = self.search.lock().await.as_ref() {
            imp.search_in_language(git_url, language, tokens, limit, offset)
                .await
        } else {
            Err(CodeSearchError::NotReady)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closest_match() {
        // Test .git suffix should still match
        assert_eq!(
            closest_match(
                "https://github.com/example/test.git",
                [&RepositoryConfig::new(
                    "https://github.com/example/test".to_string()
                )]
            ),
            Some("https://github.com/example/test".into())
        );

        // Test auth in URL should still match
        assert_eq!(
            closest_match(
                "https://creds@github.com/example/test",
                [&RepositoryConfig::new(
                    "https://github.com/example/test".to_string()
                )]
            ),
            Some("https://github.com/example/test".into())
        );

        // Test name must be exact match
        assert_eq!(
            closest_match(
                "https://github.com/example/another-repo",
                [&RepositoryConfig::new(
                    "https://github.com/example/anoth-repo".to_string()
                )]
            ),
            None
        );

        // Test different repositories with a common prefix should not match
        assert_eq!(
            closest_match(
                "https://github.com/TabbyML/tabby",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/registry-tabby".to_string()
                )]
            ),
            None
        );

        // Test entirely different repository names should not match
        assert_eq!(
            closest_match(
                "https://github.com/TabbyML/tabby",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/uptime".to_string()
                )]
            ),
            None
        );

        assert_eq!(
            closest_match(
                "https://github.com",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/tabby".to_string()
                )],
            ),
            None
        );

        // Test different host
        assert_eq!(
            closest_match(
                "https://bitbucket.com/TabbyML/tabby",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/tabby".to_string()
                )]
            ),
            Some("https://github.com/TabbyML/tabby".into())
        );

        // Test multiple close matches
        assert_eq!(
            closest_match(
                "git@github.com:TabbyML/tabby",
                [
                    &RepositoryConfig::new("https://bitbucket.com/CrabbyML/crabby".into()),
                    &RepositoryConfig::new("https://gitlab.com/TabbyML/registry-tabby".into()),
                ],
            ),
            None
        );
    }

    #[test]
    fn test_closest_match_url_format_differences() {
        // Test different protocol and suffix should still match
        assert_eq!(
            closest_match(
                "git@github.com:TabbyML/tabby.git",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/tabby".to_string()
                )]
            ),
            Some("https://github.com/TabbyML/tabby".into())
        );

        // Test different protocol should still match
        assert_eq!(
            closest_match(
                "git@github.com:TabbyML/tabby",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/tabby".to_string()
                )]
            ),
            Some("https://github.com/TabbyML/tabby".into())
        );

        // Test URL without organization should still match
        assert_eq!(
            closest_match(
                "https://custom-git.com/tabby",
                [&RepositoryConfig::new(
                    "https://custom-git.com/TabbyML/tabby".to_string()
                )]
            ),
            Some("https://custom-git.com/TabbyML/tabby".into())
        );
    }
}
