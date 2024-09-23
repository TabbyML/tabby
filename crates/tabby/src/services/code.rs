use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use tabby_common::{
    api::code::{
        CodeSearch, CodeSearchDocument, CodeSearchError, CodeSearchHit, CodeSearchParams,
        CodeSearchQuery, CodeSearchResponse, CodeSearchScores,
    },
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

use super::tantivy::IndexReaderProvider;

struct CodeSearchImpl {
    embedding: Arc<dyn Embedding>,
}

impl CodeSearchImpl {
    fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self { embedding }
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
        query: CodeSearchQuery,
        params: CodeSearchParams,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        let docs_from_embedding = {
            let embedding = self.embedding.embed(&query.content).await?;
            let embedding_tokens_query = Box::new(index::embedding_tokens_query(
                embedding.len(),
                embedding.iter(),
            ));

            let query = code::code_search_query(&query, embedding_tokens_query);
            self.search_with_query(reader, &query, params.num_to_score)
                .await?
        };

        let docs_from_bm25 = {
            let body_tokens = tokenize_code(&query.content);
            let body_query = code::body_query(&body_tokens);

            let query = code::code_search_query(&query, body_query);
            self.search_with_query(reader, &query, params.num_to_score)
                .await?
        };

        Ok(merge_code_responses_by_rank(
            &params,
            docs_from_embedding,
            docs_from_bm25,
        ))
    }
}

const RANK_CONSTANT: f32 = 60.0;

fn merge_code_responses_by_rank(
    params: &CodeSearchParams,
    embedding_resp: Vec<(f32, TantivyDocument)>,
    bm25_resp: Vec<(f32, TantivyDocument)>,
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
        if let Some((score, _)) = scored_hits.get_mut(chunk_id) {
            score.rrf += rank;
            score.bm25 = bm25;
        } else {
            let scores = CodeSearchScores {
                rrf: rank,
                bm25,
                ..Default::default()
            };
            scored_hits.insert(chunk_id.to_owned(), (scores, doc));
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
                hit.scores.bm25 > params.min_bm25_score
                    && hit.scores.embedding > params.min_embedding_score
                    && hit.scores.rrf > params.min_rrf_score
            })
            .take(params.num_to_return)
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

struct CodeSearchService {
    imp: CodeSearchImpl,
    provider: Arc<IndexReaderProvider>,
}

impl CodeSearchService {
    pub fn new(embedding: Arc<dyn Embedding>, provider: Arc<IndexReaderProvider>) -> Self {
        Self {
            imp: CodeSearchImpl::new(embedding),
            provider,
        }
    }
}

pub fn create_code_search(
    embedding: Arc<dyn Embedding>,
    provider: Arc<IndexReaderProvider>,
) -> impl CodeSearch {
    CodeSearchService::new(embedding, provider)
}

#[async_trait]
impl CodeSearch for CodeSearchService {
    async fn search_in_language(
        &self,
        query: CodeSearchQuery,
        params: CodeSearchParams,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        if let Some(reader) = self.provider.reader().await.as_ref() {
            self.imp.search_in_language(reader, query, params).await
        } else {
            Err(CodeSearchError::NotReady)
        }
    }
}
