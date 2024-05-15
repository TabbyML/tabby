use async_trait::async_trait;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use tantivy::{
    schema::{
        self,
        document::{
            DeserializeError, DocumentDeserialize, ReferenceValue, ReferenceValueLeaf,
            ValueDeserialize,
        },
        FieldValue, OwnedValue, Value,
    },
    Document,
};
use thiserror::Error;
use utoipa::ToSchema;

use crate::index::CodeSearchSchema;

#[derive(Default, Serialize, Deserialize, Debug, ToSchema)]
pub struct CodeSearchResponse {
    pub num_hits: usize,
    pub hits: Vec<CodeSearchHit>,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct CodeSearchHit {
    pub score: f32,
    pub doc: CodeSearchDocument,
    pub id: u32,
}

#[derive(Serialize, Deserialize, Debug, Builder, ToSchema)]
pub struct CodeSearchDocument {
    /// Unique identifier for the file in the repository, stringified SourceFileKey.
    ///
    /// Skipped in API responses.
    #[serde(skip_serializing)]
    pub file_id: String,

    pub body: String,
    pub filepath: String,
    pub git_url: String,
    pub language: String,
}

#[derive(Error, Debug)]
pub enum CodeSearchError {
    #[error("index not ready")]
    NotReady,

    #[error(transparent)]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error(transparent)]
    TantivyError(#[from] tantivy::TantivyError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[async_trait]
pub trait CodeSearch: Send + Sync {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError>;

    async fn search_in_language(
        &self,
        git_url: &str,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError>;
}

impl Document for CodeSearchDocument {
    type Value<'a> = Str<'a>;
    type FieldsValuesIter<'a> = CodeSearchDocumentFieldValueIter<'a>;

    fn iter_fields_and_values(&self) -> Self::FieldsValuesIter<'_> {
        CodeSearchDocumentFieldValueIter::new(self)
    }
}

pub struct CodeSearchDocumentFieldValueIter<'a> {
    field_id: i32,
    doc: &'a CodeSearchDocument,
}

impl<'a> CodeSearchDocumentFieldValueIter<'a> {
    fn new(doc: &'a CodeSearchDocument) -> Self {
        Self { field_id: 0, doc }
    }
}

impl<'a> Iterator for CodeSearchDocumentFieldValueIter<'a> {
    type Item = (tantivy::schema::Field, Str<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let schema = CodeSearchSchema::instance();
        let item = match self.field_id {
            0 => Some((schema.field_body, Str(&self.doc.body))),
            1 => Some((schema.field_filepath, Str(&self.doc.filepath))),
            2 => Some((schema.field_git_url, Str(&self.doc.git_url))),
            3 => Some((schema.field_language, Str(&self.doc.language))),
            4 => Some((schema.field_file_id, Str(&self.doc.file_id))),
            _ => None,
        };

        if item.is_some() {
            self.field_id += 1;
        }

        item
    }
}

#[derive(Clone, Debug)]
pub struct Str<'a>(&'a str);

impl<'a> Value<'a> for Str<'a> {
    type ArrayIter = std::iter::Empty<Self>;
    type ObjectIter = std::iter::Empty<(&'a str, Self)>;

    fn as_value(&self) -> tantivy::schema::document::ReferenceValue<'a, Self> {
        ReferenceValue::Leaf(ReferenceValueLeaf::Str(&self.0))
    }
}

impl DocumentDeserialize for CodeSearchDocument {
    fn deserialize<'de, D>(mut deserializer: D) -> Result<Self, DeserializeError>
    where
        D: tantivy::schema::document::DocumentDeserializer<'de>,
    {
        let code = CodeSearchSchema::instance();
        let mut builder = CodeSearchDocumentBuilder::default();
        while let Some((field, value)) = deserializer.next_field::<String>()? {
            if field.field_id() == code.field_body.field_id() {
                builder.body(value);
            } else if field.field_id() == code.field_filepath.field_id() {
                builder.filepath(value);
            } else if field.field_id() == code.field_git_url.field_id() {
                builder.git_url(value);
            } else if field.field_id() == code.field_language.field_id() {
                builder.language(value);
            } else if field.field_id() == code.field_file_id.field_id() {
                builder.file_id(value);
            }
        }

        builder
            .build()
            .map_err(|e| DeserializeError::Custom(e.to_string()))
    }
}
