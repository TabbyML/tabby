use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use cached::{CachedAsync, TimedCache};
use parse_git_url::GitUrl;
use tabby_common::{
    api::code::{
        CodeSearch, CodeSearchDocument, CodeSearchError, CodeSearchHit, CodeSearchQuery,
        CodeSearchResponse, CodeSearchScores,
    },
    config::{ConfigAccess, RepositoryConfig},
    index::{
        self,
        code::{self, tokenize_code},
        IndexSchema,
    },
};
use tabby_inference::Embedding;
use tantivy::{
    collector::TopDocs,
    schema::{self, Value},
    IndexReader, TantivyDocument,
};
use tokio::sync::Mutex;
use tracing::debug;

use super::tantivy::IndexReaderProvider;

struct CodeSearchImpl {
    config_access: Arc<dyn ConfigAccess>,
    embedding: Arc<dyn Embedding>,
    repo_cache: Mutex<TimedCache<(), Vec<RepositoryConfig>>>,
}

impl CodeSearchImpl {
    fn new(config_access: Arc<dyn ConfigAccess>, embedding: Arc<dyn Embedding>) -> Self {
        Self {
            config_access,
            embedding,
            repo_cache: Mutex::new(TimedCache::with_lifespan(10 * 60)),
        }
    }

    async fn search_with_query(
        &self,
        reader: &IndexReader,
        q: &dyn tantivy::query::Query,
        limit: usize,
    ) -> Result<Vec<(f32, TantivyDocument)>, CodeSearchError> {
        let searcher = reader.searcher();
        let top_docs = { searcher.search(q, &(TopDocs::with_limit(limit)))? };
        let top_docs = top_docs
            .iter()
            .map(|(score, doc_address)| {
                let doc: TantivyDocument = searcher.doc(*doc_address).unwrap();
                (*score, doc)
            })
            .collect();
        Ok(top_docs)
    }

    async fn search_in_language(
        &self,
        reader: &IndexReader,
        mut query: CodeSearchQuery,
        limit: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        let mut cache = self.repo_cache.lock().await;

        let repos = cache
            .try_get_or_set_with((), || async {
                let repos = self.config_access.repositories().await?;
                Ok::<_, anyhow::Error>(repos)
            })
            .await?;

        let Some(git_url) = closest_match(&query.git_url, repos.iter()) else {
            return Ok(CodeSearchResponse::default());
        };

        debug!(
            "query.git_url: {:?}, matched git_url: {:?}",
            query.git_url, git_url
        );

        query.git_url = git_url.to_owned();

        let docs_from_embedding = {
            let embedding = self.embedding.embed(&query.content).await?;
            let embedding_tokens_query = Box::new(index::embedding_tokens_query(
                embedding.len(),
                embedding.iter(),
            ));

            let query = code::code_search_query(&query, embedding_tokens_query);
            self.search_with_query(reader, &query, limit * 2).await?
        };

        let docs_from_bm25 = {
            let body_tokens = tokenize_code(&query.content);
            let body_query = code::body_query(&body_tokens);

            let query = code::code_search_query(&query, body_query);
            self.search_with_query(reader, &query, limit * 2).await?
        };

        Ok(merge_code_responses_by_rank(
            docs_from_embedding,
            docs_from_bm25,
            limit,
        ))
    }
}

const RANK_CONSTANT: f32 = 60.0;
const EMBEDDING_SCORE_THRESHOLD: f32 = 0.75;
const BM25_SCORE_THRESHOLD: f32 = 8.0;
const RRF_SCORE_THRESHOLD: f32 = 0.028;

fn merge_code_responses_by_rank(
    embedding_resp: Vec<(f32, TantivyDocument)>,
    bm25_resp: Vec<(f32, TantivyDocument)>,
    limit: usize,
) -> CodeSearchResponse {
    let mut scored_hits: HashMap<String, (CodeSearchScores, TantivyDocument)> = HashMap::default();

    for (rank, embedding, doc) in compute_rank_score(embedding_resp).into_iter() {
        let scores = CodeSearchScores {
            rrf: rank,
            embedding,
            ..Default::default()
        };

        scored_hits.insert(get_chunk_id(&doc).to_owned(), (scores, doc));
    }

    for (rank, bm25, doc) in compute_rank_score(bm25_resp).into_iter() {
        let chunk_id = get_chunk_id(&doc);
        // Only keep items with embedding score > 0.
        if let Some((score, _)) = scored_hits.get_mut(chunk_id) {
            score.rrf += rank;
            score.bm25 = bm25;
        }
    }

    let mut scored_hits: Vec<CodeSearchHit> = scored_hits
        .into_values()
        .map(|(scores, doc)| create_hit(scores, doc))
        .collect();
    scored_hits.sort_by(|a, b| b.scores.rrf.total_cmp(&a.scores.rrf));
    CodeSearchResponse {
        hits: scored_hits
            .into_iter()
            .filter(|hit| {
                hit.scores.bm25 > BM25_SCORE_THRESHOLD
                    && hit.scores.embedding > EMBEDDING_SCORE_THRESHOLD
                    && hit.scores.rrf > RRF_SCORE_THRESHOLD
            })
            .take(limit)
            .collect(),
    }
}

fn compute_rank_score(resp: Vec<(f32, TantivyDocument)>) -> Vec<(f32, f32, TantivyDocument)> {
    resp.into_iter()
        .enumerate()
        .map(|(rank, (score, doc))| (1.0 / (RANK_CONSTANT + (rank + 1) as f32), score, doc))
        .collect()
}

fn get_chunk_id(doc: &TantivyDocument) -> &str {
    let schema = IndexSchema::instance();
    get_text(doc, schema.field_chunk_id)
}

fn create_hit(scores: CodeSearchScores, doc: TantivyDocument) -> CodeSearchHit {
    let schema = IndexSchema::instance();
    let doc = CodeSearchDocument {
        file_id: get_text(&doc, schema.field_id).to_owned(),
        chunk_id: get_text(&doc, schema.field_chunk_id).to_owned(),
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
    CodeSearchHit { scores, doc }
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
        config_access: Arc<dyn ConfigAccess>,
        embedding: Arc<dyn Embedding>,
        provider: Arc<IndexReaderProvider>,
    ) -> Self {
        Self {
            imp: CodeSearchImpl::new(config_access, embedding),
            provider,
        }
    }
}

pub fn create_code_search(
    config_access: Arc<dyn ConfigAccess>,
    embedding: Arc<dyn Embedding>,
    provider: Arc<IndexReaderProvider>,
) -> impl CodeSearch {
    CodeSearchService::new(config_access, embedding, provider)
}

#[async_trait]
impl CodeSearch for CodeSearchService {
    async fn search_in_language(
        &self,
        query: CodeSearchQuery,
        limit: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        if let Some(reader) = self.provider.reader().await.as_ref() {
            self.imp.search_in_language(reader, query, limit).await
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

    #[test]
    fn test_closest_match_local_url() {
        assert_match_first!(
            "git@github.com:TabbyML/tabby.git",
            ["file:///home/TabbyML/tabby"]
        );
    }
}
