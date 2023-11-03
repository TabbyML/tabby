use std::{sync::Arc, time::Duration};

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
    query::{QueryParser, TermQuery, TermSetQuery},
    schema::{Field, IndexRecordOption},
    DocAddress, Document, Index, IndexReader, Term,
};
use thiserror::Error;
use tokio::{sync::OnceCell, task, time::sleep};
use tracing::{debug, instrument, log::info, warn};
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
        (status = 501, description = "When code search is not enabled, the endpoint will returns 501 Not Implemented"),
        )
    )]
#[instrument(skip(state, query))]
pub async fn search(
    State(state): State<Arc<IndexServer>>,
    query: Query<SearchQuery>,
) -> Result<Json<SearchResponse>, StatusCode> {
    match state.search(
        &query.q,
        query.limit.unwrap_or(20),
        query.offset.unwrap_or(0),
    ) {
        Ok(serp) => Ok(Json(serp)),
        Err(IndexServerError::NotReady) => Err(StatusCode::NOT_IMPLEMENTED),
        Err(IndexServerError::TantivyError(err)) => {
            warn!("{}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

struct IndexServerImpl {
    reader: IndexReader,
    query_parser: QueryParser,

    field_body: Field,
    field_filepath: Field,
    field_git_url: Field,
    field_kind: Field,
    field_language: Field,
    field_name: Field,
}

impl IndexServerImpl {
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
        let query = self.query_parser.parse_query(q)?;
        self.search_with_query(&query, limit, offset)
    }

    pub fn search_with_query(
        &self,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> tantivy::Result<SearchResponse> {
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

static IMPL: OnceCell<IndexServerImpl> = OnceCell::const_new();

pub struct IndexServer {}

impl IndexServer {
    pub fn new() -> Self {
        task::spawn(IMPL.get_or_init(|| async {
            task::spawn(IndexServer::worker())
                .await
                .expect("Failed to create IndexServerImpl")
        }));
        Self {}
    }

    fn with_impl<T, F>(&self, op: F) -> Result<T, IndexServerError>
    where
        F: FnOnce(&IndexServerImpl) -> Result<T, IndexServerError>,
    {
        if let Some(imp) = IMPL.get() {
            op(imp)
        } else {
            Err(IndexServerError::NotReady)
        }
    }

    async fn worker() -> IndexServerImpl {
        loop {
            match IndexServerImpl::load() {
                Ok(index_server) => {
                    info!("Index is ready, enabling server...");
                    return index_server;
                }
                Err(err) => {
                    debug!("Source code index is not ready `{}`", err);
                }
            };

            sleep(Duration::from_secs(60)).await;
        }
    }

    pub fn language_query(&self, language: &str) -> Result<Box<TermQuery>, IndexServerError> {
        self.with_impl(|imp| {
            Ok(Box::new(TermQuery::new(
                Term::from_field_text(imp.field_language, language),
                IndexRecordOption::WithFreqsAndPositions,
            )))
        })
    }

    pub fn body_query(&self, tokens: &[String]) -> Result<Box<TermSetQuery>, IndexServerError> {
        self.with_impl(|imp| {
            Ok(Box::new(TermSetQuery::new(
                tokens
                    .iter()
                    .map(|x| Term::from_field_text(imp.field_body, x)),
            )))
        })
    }

    pub fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, IndexServerError> {
        self.with_impl(|imp| Ok(imp.search(q, limit, offset)?))
    }

    pub fn search_with_query(
        &self,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, IndexServerError> {
        self.with_impl(|imp| Ok(imp.search_with_query(q, limit, offset)?))
    }
}

#[derive(Error, Debug)]
pub enum IndexServerError {
    #[error("index not ready")]
    NotReady,

    #[error("{0}")]
    TantivyError(#[from] tantivy::TantivyError),
}
