use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    Json,
};
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tabby_common::{index::IndexExt, path};
use tantivy::{
    collector::{Count, TopDocs},
    query::QueryParser,
    schema::{Field, FieldType, NamedFieldDocument, Schema},
    DocAddress, Document, Index, IndexReader, Score,
};
use tracing::instrument;
use utoipa::{IntoParams, ToSchema};

#[derive(Deserialize, IntoParams)]
pub struct SearchQuery {
    #[param(default = "get")]
    q: String,

    #[param(default = 20)]
    limit: Option<usize>,

    #[param(default = 0)]
    offset: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchResponse {
    q: String,
    num_hits: usize,
    hits: Vec<Hit>,
}

#[derive(Serialize)]
pub struct Hit {
    score: Score,
    doc: NamedFieldDocument,
    id: u32,
}

#[utoipa::path(
    get,
    params(SearchQuery),
    path = "/v1beta/search",
    operation_id = "search",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success" , content_type = "application/json"),
        (status = 405, description = "When code search is not enabled, the endpoint will returns 405 Method Not Allowed"),
    )
)]
#[instrument(skip(state, query))]
pub async fn search(
    State(state): State<Arc<IndexServer>>,
    query: Query<SearchQuery>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let Ok(serp) = state.search(
        &query.q,
        query.limit.unwrap_or(20),
        query.offset.unwrap_or(0),
    ) else {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    };

    Ok(Json(serp))
}

pub struct IndexServer {
    reader: IndexReader,
    query_parser: QueryParser,
    schema: Schema,
}

impl IndexServer {
    pub fn new() -> Self {
        Self::load().expect("Failed to load code state")
    }

    fn load() -> Result<Self> {
        let index = Index::open_in_dir(path::index_dir())?;
        index.register_tokenizer();

        let schema = index.schema();
        let default_fields: Vec<Field> = schema
            .fields()
            .filter(|&(_, field_entry)| match field_entry.field_type() {
                FieldType::Str(ref text_field_options) => {
                    text_field_options.get_indexing_options().is_some()
                }
                _ => false,
            })
            .map(|(field, _)| field)
            .collect();
        let query_parser =
            QueryParser::new(schema.clone(), default_fields, index.tokenizers().clone());
        let reader = index.reader()?;
        Ok(Self {
            reader,
            query_parser,
            schema,
        })
    }

    fn search(&self, q: &str, limit: usize, offset: usize) -> tantivy::Result<SearchResponse> {
        let query = self
            .query_parser
            .parse_query(q)
            .expect("Parsing the query failed");
        let searcher = self.reader.searcher();
        let (top_docs, num_hits) = {
            searcher.search(
                &query,
                &(TopDocs::with_limit(limit).and_offset(offset), Count),
            )?
        };
        let hits: Vec<Hit> = {
            top_docs
                .iter()
                .map(|(score, doc_address)| {
                    let doc = searcher.doc(*doc_address).unwrap();
                    self.create_hit(*score, doc, *doc_address)
                })
                .collect()
        };
        Ok(SearchResponse {
            q: q.to_owned(),
            num_hits,
            hits,
        })
    }

    fn create_hit(&self, score: Score, doc: Document, doc_address: DocAddress) -> Hit {
        Hit {
            score,
            doc: self.schema.to_named_doc(&doc),
            id: doc_address.doc_id,
        }
    }
}
