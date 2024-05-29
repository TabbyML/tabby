use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use cached::{CachedAsync, TimedCache};
use parse_git_url::GitUrl;
use tabby_common::{
    api::code::{
        CodeSearch, CodeSearchDocument, CodeSearchError, CodeSearchHit, CodeSearchQuery,
        CodeSearchResponse,
    },
    config::{RepositoryAccess, RepositoryConfig},
    index::{code, IndexSchema},
};
use tantivy::{
    collector::{Count, TopDocs},
    schema::{self, Value},
    IndexReader, TantivyDocument,
};
use tokio::sync::Mutex;

use super::tantivy::IndexReaderProvider;

struct CodeSearchImpl {
    repository_access: Arc<dyn RepositoryAccess>,
    repo_cache: Mutex<TimedCache<(), Vec<RepositoryConfig>>>,
}

impl CodeSearchImpl {
    fn new(repository_access: Arc<dyn RepositoryAccess>) -> Self {
        Self {
            repository_access,
            repo_cache: Mutex::new(TimedCache::with_lifespan(10 * 60)),
        }
    }

    fn create_hit(&self, score: f32, doc: TantivyDocument) -> CodeSearchHit {
        let schema = IndexSchema::instance();
        let doc = CodeSearchDocument {
            file_id: get_text(&doc, schema.field_id).to_owned(),
            body: get_json_text_field(
                &doc,
                schema.field_chunk_attributes,
                code::fields::CHUNK_BODY,
            )
            .to_owned(),
            filepath: get_json_text_field(
                &doc,
                schema.field_chunk_attributes,
                code::fields::CHUNK_FILEPATH,
            )
            .to_owned(),
            git_url: get_json_text_field(
                &doc,
                schema.field_chunk_attributes,
                code::fields::CHUNK_GIT_URL,
            )
            .to_owned(),
            language: get_json_text_field(
                &doc,
                schema.field_chunk_attributes,
                code::fields::CHUNK_LANGUAGE,
            )
            .to_owned(),
            start_line: get_json_number_field(
                &doc,
                schema.field_chunk_attributes,
                code::fields::CHUNK_START_LINE,
            ) as usize,
        };
        CodeSearchHit { score, doc }
    }

    async fn search_with_query(
        &self,
        reader: &IndexReader,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        let searcher = reader.searcher();
        let (top_docs, num_hits) =
            { searcher.search(q, &(TopDocs::with_limit(limit).and_offset(offset), Count))? };
        let hits: Vec<CodeSearchHit> = {
            top_docs
                .iter()
                .map(|(score, doc_address)| {
                    let doc: TantivyDocument = searcher.doc(*doc_address).unwrap();
                    self.create_hit(*score, doc)
                })
                .collect()
        };
        Ok(CodeSearchResponse { num_hits, hits })
    }

    async fn search_in_language(
        &self,
        reader: &IndexReader,
        mut query: CodeSearchQuery,
        limit: usize,
        offset: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        let mut cache = self.repo_cache.lock().await;

        let repos = cache
            .try_get_or_set_with((), || async {
                let repos = self.repository_access.list_repositories().await?;
                Ok::<_, anyhow::Error>(repos)
            })
            .await?;

        let Some(git_url) = closest_match(&query.git_url, repos.iter()) else {
            return Ok(CodeSearchResponse::default());
        };

        query.git_url = git_url.to_owned();

        let query = code::code_search_query(&query);
        self.search_with_query(reader, &query, limit, offset).await
    }
}

fn get_text(doc: &TantivyDocument, field: schema::Field) -> &str {
    doc.get_first(field).unwrap().as_str().unwrap()
}

fn get_json_number_field(doc: &TantivyDocument, field: schema::Field, name: &str) -> i64 {
    doc.get_first(field)
        .unwrap()
        .as_object()
        .unwrap()
        .find(|(k, _)| *k == name)
        .unwrap()
        .1
        .as_i64()
        .unwrap()
}

fn get_json_text_field<'a>(doc: &'a TantivyDocument, field: schema::Field, name: &str) -> &'a str {
    doc.get_first(field)
        .unwrap()
        .as_object()
        .unwrap()
        .find(|(k, _)| *k == name)
        .unwrap()
        .1
        .as_str()
        .unwrap()
}

fn closest_match<'a>(
    search_term: &'a str,
    search_input: impl IntoIterator<Item = &'a RepositoryConfig>,
) -> Option<&'a str> {
    let git_search = GitUrl::parse(search_term).ok()?;

    search_input
        .into_iter()
        .filter(|elem| GitUrl::parse(&elem.git_url).is_ok_and(|x| x.name == git_search.name))
        // If there're multiple matches, we pick the one with highest alphabetical order
        .min_by_key(|elem| elem.canonical_git_url())
        .map(|x| x.git_url.as_str())
}

struct CodeSearchService {
    imp: CodeSearchImpl,
    provider: Arc<IndexReaderProvider>,
}

impl CodeSearchService {
    pub fn new(
        repository_access: Arc<dyn RepositoryAccess>,
        provider: Arc<IndexReaderProvider>,
    ) -> Self {
        Self {
            imp: CodeSearchImpl::new(repository_access),
            provider,
        }
    }
}

pub fn create_code_search(
    repository_access: Arc<dyn RepositoryAccess>,
    provider: Arc<IndexReaderProvider>,
) -> impl CodeSearch {
    CodeSearchService::new(repository_access, provider)
}

#[async_trait]
impl CodeSearch for CodeSearchService {
    async fn search_in_language(
        &self,
        query: CodeSearchQuery,
        limit: usize,
        offset: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        if let Some(reader) = self.provider.reader().await.as_ref() {
            self.imp
                .search_in_language(reader, query, limit, offset)
                .await
        } else {
            Err(CodeSearchError::NotReady)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_match_first {
        ($query:literal, $candidates:expr) => {
            let candidates: Vec<_> = $candidates
                .into_iter()
                .map(|x| RepositoryConfig::new(x.to_string()))
                .collect();
            let expect = &candidates[0];
            assert_eq!(
                closest_match($query, &candidates),
                Some(expect.git_url.as_ref())
            );
        };
    }

    macro_rules! assert_match_none {
        ($query:literal, $candidates:expr) => {
            let candidates: Vec<_> = $candidates
                .into_iter()
                .map(|x| RepositoryConfig::new(x.to_string()))
                .collect();
            assert_eq!(closest_match($query, &candidates), None);
        };
    }

    #[test]
    fn test_closest_match() {
        // Test .git suffix should still match
        assert_match_first!(
            "https://github.com/example/test.git",
            ["https://github.com/example/test"]
        );

        // Test auth in URL should still match
        assert_match_first!(
            "https://creds@github.com/example/test",
            ["https://github.com/example/test"]
        );

        // Test name must be exact match
        assert_match_none!(
            "https://github.com/example/another-repo",
            ["https://github.com/example/anoth-repo"]
        );

        // Test different repositories with a common prefix should not match
        assert_match_none!(
            "https://github.com/TabbyML/tabby",
            ["https://github.com/TabbyML/registry-tabby"]
        );

        // Test entirely different repository names should not match
        assert_match_none!(
            "https://github.com/TabbyML/tabby",
            ["https://github.com/TabbyML/uptime"]
        );

        assert_match_none!("https://github.com", ["https://github.com/TabbyML/tabby"]);

        // Test different host
        assert_match_first!(
            "https://bitbucket.com/TabbyML/tabby",
            ["https://github.com/TabbyML/tabby"]
        );

        // Test multiple close matches
        assert_match_none!(
            "git@github.com:TabbyML/tabby",
            [
                "https://bitbucket.com/CrabbyML/crabby",
                "https://gitlab.com/TabbyML/registry-tabby",
            ]
        );
    }

    #[test]
    fn test_closest_match_url_format_differences() {
        // Test different protocol and suffix should still match
        assert_match_first!(
            "git@github.com:TabbyML/tabby.git",
            ["https://github.com/TabbyML/tabby"]
        );

        // Test different protocol should still match
        assert_match_first!(
            "git@github.com:TabbyML/tabby",
            ["https://github.com/TabbyML/tabby"]
        );

        // Test URL without organization should still match
        assert_match_first!(
            "https://custom-git.com/tabby",
            ["https://custom-git.com/TabbyML/tabby"]
        );
    }
}
