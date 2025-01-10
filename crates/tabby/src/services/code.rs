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
        corpus, IndexSchema,
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

        Ok(
            merge_code_responses_by_rank(reader, &params, docs_from_embedding, docs_from_bm25)
                .await,
        )
    }
}

const RANK_CONSTANT: f32 = 60.0;

async fn merge_code_responses_by_rank(
    reader: &IndexReader,
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

    let scored_hits_futures: Vec<_> = scored_hits
        .into_values()
        .map(|(scores, doc)| create_hit(reader, scores, doc))
        .collect();
    let mut scored_hits: Vec<CodeSearchHit> = futures::future::join_all(scored_hits_futures)
        .await
        .into_iter()
        .collect();
    scored_hits.sort_by(|a, b| b.scores.rrf.total_cmp(&a.scores.rrf));
    retain_at_most_two_hits_per_file(&mut scored_hits);

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

fn retain_at_most_two_hits_per_file(scored_hits: &mut Vec<CodeSearchHit>) {
    let mut scored_hits_by_fileid: HashMap<String, usize> = HashMap::default();
    scored_hits.retain(|x| {
        let count: usize = scored_hits_by_fileid
            .get(&x.doc.file_id)
            .copied()
            .unwrap_or_default();
        scored_hits_by_fileid.insert(x.doc.file_id.clone(), count + 1);
        count < 2
    });
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

async fn create_hit(
    reader: &IndexReader,
    scores: CodeSearchScores,
    doc: TantivyDocument,
) -> CodeSearchHit {
    let schema = IndexSchema::instance();
    let file_id = get_text(&doc, schema.field_id).to_owned();
    let commit = get_commit(reader, &file_id).await;

    let doc = CodeSearchDocument {
        file_id,
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
        // commit is introduced in v0.23, but it is also a required field
        // so we need to handle the case where it's not present
        commit,
        language: get_json_text_field(
            &doc,
            schema.field_chunk_attributes,
            code::fields::CHUNK_LANGUAGE,
        )
        .to_owned(),
        start_line: get_optional_json_number_field(
            &doc,
            schema.field_chunk_attributes,
            code::fields::CHUNK_START_LINE,
        ),
    };
    CodeSearchHit { scores, doc }
}

async fn get_commit(reader: &IndexReader, id: &str) -> Option<String> {
    let schema = IndexSchema::instance();
    let query = schema.doc_query(corpus::CODE, id);
    let doc = reader
        .searcher()
        .search(&query, &TopDocs::with_limit(1))
        .ok()?;
    if doc.is_empty() {
        return None;
    }

    let doc = reader.searcher().doc(doc[0].1).ok()?;
    get_json_text_field_optional(&doc, schema.field_attributes, code::fields::COMMIT)
        .map(|s| s.to_owned())
}

fn get_text(doc: &TantivyDocument, field: schema::Field) -> &str {
    doc.get_first(field).unwrap().as_str().unwrap()
}

fn get_optional_json_number_field(
    doc: &TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<usize> {
    doc.get_first(field)
        .unwrap()
        .as_object()
        .unwrap()
        .find(|(k, _)| *k == name)?
        .1
        .as_i64()
        .map(|x| x as usize)
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

fn get_json_text_field_optional<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<&'a str> {
    doc.get_first(field)
        .and_then(|value| value.as_object())
        .and_then(|mut obj| obj.find(|(k, _)| *k == name))
        .and_then(|(_, v)| v.as_str())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retain_at_most_two_hits_per_file() {
        let new_hit = |file: &str, body: &str| CodeSearchHit {
            scores: Default::default(),
            doc: CodeSearchDocument {
                file_id: file.to_string(),
                chunk_id: "chunk1".to_owned(),
                body: body.to_string(),
                filepath: "".to_owned(),
                git_url: "".to_owned(),
                commit: Some("".to_owned()),
                language: "".to_owned(),
                start_line: Some(0),
            },
        };

        let cases = vec![
            (vec![], vec![]),
            (
                vec![new_hit("file1", "body1")],
                vec![new_hit("file1", "body1")],
            ),
            (
                vec![new_hit("file1", "body1"), new_hit("file1", "body2")],
                vec![new_hit("file1", "body1"), new_hit("file1", "body2")],
            ),
            (
                vec![
                    new_hit("file1", "body1"),
                    new_hit("file1", "body2"),
                    new_hit("file1", "body3"),
                ],
                vec![new_hit("file1", "body1"), new_hit("file1", "body2")],
            ),
            (
                vec![
                    new_hit("file1", "body1"),
                    new_hit("file1", "body2"),
                    new_hit("file1", "body3"),
                    new_hit("file2", "body4"),
                ],
                vec![
                    new_hit("file1", "body1"),
                    new_hit("file1", "body2"),
                    new_hit("file2", "body4"),
                ],
            ),
            (
                vec![
                    new_hit("file1", "body1"),
                    new_hit("file1", "body2"),
                    new_hit("file1", "body3"),
                    new_hit("file2", "body4"),
                    new_hit("file2", "body5"),
                ],
                vec![
                    new_hit("file1", "body1"),
                    new_hit("file1", "body2"),
                    new_hit("file2", "body4"),
                    new_hit("file2", "body5"),
                ],
            ),
            (
                vec![
                    new_hit("file1", "body1"),
                    new_hit("file1", "body2"),
                    new_hit("file1", "body3"),
                    new_hit("file2", "body4"),
                    new_hit("file2", "body5"),
                    new_hit("file2", "body6"),
                ],
                vec![
                    new_hit("file1", "body1"),
                    new_hit("file1", "body2"),
                    new_hit("file2", "body4"),
                    new_hit("file2", "body5"),
                ],
            ),
        ];

        for (input, expected) in cases {
            let mut input = input;
            retain_at_most_two_hits_per_file(&mut input);
            assert_eq!(input, expected);
        }
    }
}
