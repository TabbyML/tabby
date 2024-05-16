use tantivy::{
    schema::{
        document::{DeserializeError, DocumentDeserialize, ReferenceValue, ReferenceValueLeaf},
        OwnedValue, Value,
    },
    Document,
};

use super::CodeSearchSchema;
use crate::api::code::{CodeSearchDocument, CodeSearchDocumentBuilder};
impl Document for CodeSearchDocument {
    type Value<'a> = RefValue<'a>;
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
    type Item = (tantivy::schema::Field, RefValue<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let schema = CodeSearchSchema::instance();
        let item = match self.field_id {
            0 => Some((schema.field_body, RefValue::Str(&self.doc.body))),
            1 => Some((schema.field_filepath, RefValue::Str(&self.doc.filepath))),
            2 => Some((schema.field_git_url, RefValue::Str(&self.doc.git_url))),
            3 => Some((schema.field_language, RefValue::Str(&self.doc.language))),
            4 => Some((schema.field_file_id, RefValue::Str(&self.doc.file_id))),
            5 => Some((
                schema.field_start_line,
                RefValue::Usize(self.doc.start_line),
            )),
            _ => None,
        };

        if item.is_some() {
            self.field_id += 1;
        }

        item
    }
}

#[derive(Clone, Debug)]
pub enum RefValue<'a> {
    Str(&'a str),
    Usize(usize),
}
impl<'a> Value<'a> for RefValue<'a> {
    type ArrayIter = std::iter::Empty<Self>;
    type ObjectIter = std::iter::Empty<(&'a str, Self)>;

    fn as_value(&self) -> tantivy::schema::document::ReferenceValue<'a, Self> {
        match self {
            RefValue::Str(s) => ReferenceValue::Leaf(ReferenceValueLeaf::Str(s)),
            RefValue::Usize(i) => ReferenceValue::Leaf(ReferenceValueLeaf::U64(*i as u64)),
        }
    }
}

impl DocumentDeserialize for CodeSearchDocument {
    fn deserialize<'de, D>(mut deserializer: D) -> Result<Self, DeserializeError>
    where
        D: tantivy::schema::document::DocumentDeserializer<'de>,
    {
        let code = CodeSearchSchema::instance();
        let mut builder = CodeSearchDocumentBuilder::default();
        while let Some((field, value)) = deserializer.next_field::<OwnedValue>()? {
            let value = value.as_ref();
            if field == code.field_body {
                builder.body(value.try_deserialize()?);
            } else if field == code.field_filepath {
                builder.filepath(value.try_deserialize()?);
            } else if field == code.field_git_url {
                builder.git_url(value.try_deserialize()?);
            } else if field == code.field_language {
                builder.language(value.try_deserialize()?);
            } else if field == code.field_file_id {
                builder.file_id(value.try_deserialize()?);
            } else if field == code.field_start_line {
                builder.start_line(value.try_deserialize()?);
            }
        }

        builder
            .build()
            .map_err(|e| DeserializeError::Custom(e.to_string()))
    }
}

trait TryDeserialize<T> {
    fn try_deserialize(self) -> Result<T, DeserializeError>;
}

impl TryDeserialize<usize> for &OwnedValue {
    fn try_deserialize(self) -> Result<usize, DeserializeError> {
        match self {
            OwnedValue::U64(i) => Ok(*i as usize),
            _ => Err(DeserializeError::Custom(
                "Field type doesn't match".to_string(),
            )),
        }
    }
}

impl TryDeserialize<String> for &OwnedValue {
    fn try_deserialize(self) -> Result<String, DeserializeError> {
        match self {
            OwnedValue::Str(s) => Ok(s.to_string()),
            _ => Err(DeserializeError::Custom(
                "Field type doesn't match".to_string(),
            )),
        }
    }
}
