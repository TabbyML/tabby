use std::{
    sync::Mutex,
    time::{Duration, SystemTime},
};

use anyhow::Result;
use serde::Serialize;
use tabby_common::{index::IndexExt, path};
use tantivy::{
    collector::{Count, TopDocs},
    query::{QueryParser, TermQuery, TermSetQuery},
    schema::{Field, IndexRecordOption},
    DocAddress, Document, Index, IndexReader, Term,
};
use thiserror::Error;
use tracing::{debug, log::info};
use utoipa::ToSchema;

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

#[derive(Error, Debug)]
pub enum CodeSearchError {
    #[error("index not ready")]
    NotReady,

    #[error("{0}")]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error("{0}")]
    TantivyError(#[from] tantivy::TantivyError),
}

pub trait CodeSearch {
    fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError>;

    fn search_with_query(
        &self,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError>;
}

struct CodeSearchImpl {
    reader: IndexReader,
    query_parser: QueryParser,

    field_body: Field,
    field_filepath: Field,
    field_git_url: Field,
    field_kind: Field,
    field_language: Field,
    field_name: Field,
}

impl CodeSearchImpl {
    fn load() -> Result<Self> {
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

impl CodeSearch for CodeSearchImpl {
    fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        let query = self.query_parser.parse_query(q)?;
        self.search_with_query(&query, limit, offset)
    }

    fn search_with_query(
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

fn get_field(doc: &Document, field: Field) -> String {
    doc.get_first(field)
        .and_then(|x| x.as_text())
        .unwrap()
        .to_owned()
}

pub struct CodeSearchService {
    search: Mutex<Option<CodeSearchImpl>>,
    last_load_time: Mutex<SystemTime>,
}

impl CodeSearchService {
    pub fn new() -> Self {
        let ret = Self {
            search: Mutex::new(None),
            last_load_time: Mutex::new(SystemTime::UNIX_EPOCH),
        };
        ret.load();
        ret
    }

    fn with_impl<T, F>(&self, op: F) -> Result<T, CodeSearchError>
    where
        F: FnOnce(&CodeSearchImpl) -> Result<T, CodeSearchError>,
    {
        if let Some(imp) = self.search.lock().unwrap().as_ref() {
            op(imp)
        } else {
            self.load();
            Err(CodeSearchError::NotReady)
        }
    }

    fn load(&self) {
        let mut last_load_time = self.last_load_time.lock().unwrap();
        let duration = SystemTime::now().duration_since(*last_load_time).unwrap();

        if duration < Duration::from_secs(60) {
            return;
        }

        match CodeSearchImpl::load() {
            Ok(code) => {
                info!("Index is ready, enabling server...");
                *self.search.lock().unwrap() = Some(code)
            }
            Err(err) => {
                debug!("Source code index is not ready `{}`", err);
            }
        };

        *last_load_time = SystemTime::now();
    }

    pub fn language_query(&self, language: &str) -> Result<Box<TermQuery>, CodeSearchError> {
        self.with_impl(|imp| {
            Ok(Box::new(TermQuery::new(
                Term::from_field_text(imp.field_language, language),
                IndexRecordOption::WithFreqsAndPositions,
            )))
        })
    }

    pub fn body_query(&self, tokens: &[String]) -> Result<Box<TermSetQuery>, CodeSearchError> {
        self.with_impl(|imp| {
            Ok(Box::new(TermSetQuery::new(
                tokens
                    .iter()
                    .map(|x| Term::from_field_text(imp.field_body, x)),
            )))
        })
    }
}

impl CodeSearch for CodeSearchService {
    fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        self.with_impl(|imp| imp.search(q, limit, offset))
    }

    fn search_with_query(
        &self,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        self.with_impl(|imp| imp.search_with_query(q, limit, offset))
    }
}
