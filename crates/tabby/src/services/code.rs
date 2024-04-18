use std::{sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use cached::{CachedAsync, TimedCache};
use nucleo::Utf32String;
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
    let search_term = RepositoryConfig::canonicalize_url(search_term);

    let mut nucleo = nucleo::Matcher::new(nucleo::Config::DEFAULT.match_paths());
    search_input
        .into_iter()
        .filter_map(|entry| {
            let url = entry.canonical_git_url();
            Some((
                url.clone(),
                // Matching using the input URL as the haystack instead of the needle yielded better scoring
                // Example:
                // haystack = "https://github.com/boxbeam/untwine" needle = "https://abc@github.com/boxbeam/untwine.git" => No match
                // haystack = "https://abc@github.com/boxbeam/untwine.git" needle = "https://github.com/boxbeam/untwine" => Match, score 842
                nucleo.fuzzy_match(
                    Utf32String::from(&*search_term).slice(..),
                    Utf32String::from(&*url).slice(..),
                )?,
            ))
        })
        .max_by_key(|(_, score)| *score)
        .map(|(entry, _score)| entry)
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
        assert_eq!(
            closest_match(
                "https://github.com/example/test.git",
                [&RepositoryConfig::new(
                    "https://github.com/example/test".to_string()
                )]
            ),
            Some("https://github.com/example/test".into())
        );

        assert_eq!(
            closest_match(
                "https://creds@github.com/example/test",
                [&RepositoryConfig::new(
                    "https://github.com/example/test".to_string()
                )]
            ),
            Some("https://github.com/example/test".into())
        );

        assert_eq!(
            closest_match(
                "https://github.com/example/another-repo",
                [&RepositoryConfig::new(
                    "https://github.com/examp/anoth-repo".to_string()
                )]
            ),
            Some("https://github.com/examp/anoth-repo".into())
        );

        assert_eq!(
            closest_match(
                "https://github.com/TabbyML/tabby",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/registry-tabby".to_string()
                )]
            ),
            None
        );

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

        assert_eq!(
            closest_match(
                "https://bitbucket.com/TabbyML/tabby",
                [&RepositoryConfig::new(
                    "https://github.com/TabbyML/tabby".to_string()
                )]
            ),
            None
        );
    }
}
