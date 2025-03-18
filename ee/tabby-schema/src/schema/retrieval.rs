use juniper::GraphQLObject;
use tabby_common::api::code::{CodeSearchDocument, CodeSearchHit, CodeSearchScores};

use crate::thread::{MessageAttachmentCode, MessageAttachmentCodeFileList};

#[derive(GraphQLObject)]
pub struct AttachmentCodeHits {
    pub hits: Vec<AttachmentCodeHit>,
}

#[derive(GraphQLObject)]
pub struct AttachmentCodeHit {
    pub code: AttachmentCode,
    pub scores: AttachmentCodeScores,
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentCodeScores {
    pub rrf: f64,
    pub bm25: f64,
    pub embedding: f64,
}

#[derive(GraphQLObject, Clone, PartialEq)]
pub struct AttachmentCode {
    pub git_url: String,
    pub commit: Option<String>,
    pub filepath: String,
    pub language: String,
    pub content: String,

    /// When start line is `None`, it represents the entire file.
    pub start_line: Option<i32>,
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentCodeFileList {
    pub file_list: Vec<String>,
    pub truncated: bool,
}

impl From<&MessageAttachmentCodeFileList> for AttachmentCodeFileList {
    fn from(value: &MessageAttachmentCodeFileList) -> Self {
        Self {
            file_list: value.file_list.clone(),
            truncated: value.truncated,
        }
    }
}

impl From<AttachmentCodeFileList> for MessageAttachmentCodeFileList {
    fn from(value: AttachmentCodeFileList) -> Self {
        Self {
            file_list: value.file_list,
            truncated: value.truncated,
        }
    }
}

impl From<&MessageAttachmentCode> for AttachmentCode {
    fn from(value: &MessageAttachmentCode) -> Self {
        Self {
            git_url: value.git_url.clone(),
            commit: value.commit.clone(),
            filepath: value.filepath.clone(),
            language: value.language.clone(),
            content: value.content.clone(),
            start_line: value.start_line,
        }
    }
}

impl From<&AttachmentCode> for MessageAttachmentCode {
    fn from(value: &AttachmentCode) -> Self {
        Self {
            git_url: value.git_url.clone(),
            commit: value.commit.clone(),
            filepath: value.filepath.clone(),
            language: value.language.clone(),
            content: value.content.clone(),
            start_line: value.start_line,
        }
    }
}

impl From<CodeSearchHit> for AttachmentCodeHit {
    fn from(val: CodeSearchHit) -> Self {
        Self {
            code: val.doc.into(),
            scores: val.scores.into(),
        }
    }
}

impl From<CodeSearchDocument> for AttachmentCode {
    fn from(val: CodeSearchDocument) -> Self {
        Self {
            git_url: val.git_url.clone(),
            commit: val.commit.clone(),
            filepath: val.filepath.clone(),
            language: val.language.clone(),
            content: val.body.clone(),
            start_line: val.start_line.map(|x| x as i32),
        }
    }
}

impl From<CodeSearchScores> for AttachmentCodeScores {
    fn from(val: CodeSearchScores) -> Self {
        Self {
            rrf: val.rrf as f64,
            bm25: val.bm25 as f64,
            embedding: val.embedding as f64,
        }
    }
}
