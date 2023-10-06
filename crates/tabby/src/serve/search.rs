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
    schema::Field,
    DocAddress, Document, Index, IndexReader,
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

#[derive(Serialize, ToSchema)]
pub struct SearchResponse {
    pub num_hits: usize,
    pub hits: Vec<Hit>,
}

#[derive(Serialize, ToSchema)]
pub struct Hit {
    pub score: f32,
    pub doc: HitDocument,
    pub id: u32,
}

#[derive(Serialize, ToSchema)]
pub struct HitDocument {
    pub body: String,
    pub filepath: String,
    pub git_url: String,
    pub kind: String,
    pub language: String,
    pub name: String,
}

#[utoipa::path(
    get,
    params(SearchQuery),
    path = "/v1beta/search",
    operation_id = "search",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success" , body = SearchResponse, content_type = "application/json"),
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

    field_body: Field,
    field_filepath: Field,
    field_git_url: Field,
    field_kind: Field,
    field_language: Field,
    field_name: Field,
}

impl IndexServer {
    pub fn load() -> Result<Self> {
        let index = Index::open_in_dir(path::index_dir())?;
        index.register_tokenizer();

        let schema = index.schema();
        let field_body = schema.get_field("body").unwrap();
        let query_parser =
            QueryParser::new(schema.clone(), vec![field_body], index.tokenizers().clone());
        let reader = index
            .reader_builder()
            .reload_policy(tantivy::ReloadPolicy::OnCommit)
            .try_into()?;
        Ok(Self {
            reader,
            query_parser,
            field_body,
            field_filepath: schema.get_field("filepath").unwrap(),
            field_git_url: schema.get_field("git_url").unwrap(),
            field_kind: schema.get_field("kind").unwrap(),
            field_language: schema.get_field("language").unwrap(),
            field_name: schema.get_field("name").unwrap(),
        })
    }

    pub fn search(&self, q: &str, limit: usize, offset: usize) -> tantivy::Result<SearchResponse> {
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
        Ok(SearchResponse { num_hits, hits })
    }

    fn create_hit(&self, score: f32, doc: Document, doc_address: DocAddress) -> Hit {
        Hit {
            score,
            doc: HitDocument {
                body: get_field(&doc, self.field_body),
                filepath: get_field(&doc, self.field_filepath),
                git_url: get_field(&doc, self.field_git_url),
                kind: get_field(&doc, self.field_kind),
                name: get_field(&doc, self.field_name),
                language: get_field(&doc, self.field_language),
            },
            id: doc_address.doc_id,
        }
    }
}

fn get_field(doc: &Document, field: Field) -> String {
    doc.get_first(field)
        .and_then(|x| x.as_text())
        .unwrap()
        .to_owned()
}
