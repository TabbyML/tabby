use std::sync::Arc;

use axum::{extract::{State, Query}, Json};
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tabby_common::path::index_dir;
use tantivy::{
    collector::TopDocs,
    query::{QueryParser, QueryParserError},
    DocAddress, Index, Score, TantivyError, schema::Field, IndexReader,
};
use tracing::info;
use utoipa::{ToSchema, IntoParams};

pub struct SearchState {
    index: Index,
    reader: IndexReader,
    query_parser: QueryParser,
    git_url_field: Field,
    language_field: Field,
    content_field: Field,
}

impl SearchState {
    pub fn new() -> Option<Self> {
        let index = Index::open_in_dir(index_dir()).ok()?;
        let content_field = index.schema().get_field("content").ok()?;
        let language_field = index.schema().get_field("language").ok()?;
        let git_url_field = index.schema().get_field("git_url").ok()?;

        let query_parser = QueryParser::for_index(&index, vec![content_field]);
        let state = SearchState {
            reader: index.reader().ok()?,
            query_parser,
            content_field,
            language_field,
            git_url_field,
            index,
        };

        Some(state)
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Document {
    git_url: String,
    language: String,
    content: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct SearchResponse {
    docs: Vec<Document>,
}

#[derive(Deserialize, IntoParams)]
pub struct SearchRequest {
    #[param(example = "function")]
    q: String,
    #[param(example = "10")]
    limit: Option<usize>,
    #[param(example = "0")]
    offset: Option<usize>
}

#[utoipa::path(
    get,
    path = "/experimental/search",
    tag = "experimental",
    params(SearchRequest),
    responses(
        (status = 200, description = "Success", body = SearchResponse, content_type = "application/json"),
        (status = 501, description = "Not Implemented"),
    )
)]
pub async fn search(
    State(state): State<Arc<SearchState>>,
    params: Query<SearchRequest>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let searcher = state
        .reader
        .searcher();

    let query = state
        .query_parser
        .parse_query(&params.q)
        .map_err(QueryParserError::status)?;

    let search_options = TopDocs::with_limit(params.limit.unwrap_or(10))
        .and_offset(params.offset.unwrap_or(0));
    let top_docs: Vec<(Score, DocAddress)> = searcher
        .search(&query, &search_options)
        .map_err(TantivyError::status)?;

    let docs = top_docs
        .iter()
        .filter_map(|(_score, doc_address)| {
            let Some(retrieved_doc) = searcher.doc(*doc_address).ok() else {
                return None;
            };

            Some(Document {
                git_url: retrieved_doc.get_first(state.git_url_field).unwrap().as_text().unwrap().to_owned(),
                language: retrieved_doc.get_first(state.language_field).unwrap().as_text().unwrap().to_owned(),
                content: retrieved_doc.get_first(state.content_field).unwrap().as_text().unwrap().to_owned(),
            })
        })
        .collect();

    Ok(Json(SearchResponse { docs }))
}

trait Handler<T> {
    fn status(self) -> StatusCode;
}

impl Handler<TantivyError> for TantivyError {
    fn status(self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

impl Handler<QueryParserError> for QueryParserError {
    fn status(self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}
