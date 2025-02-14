use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Attachment {
    pub code: Option<Vec<AttachmentCode>>,
    pub client_code: Option<Vec<AttachmentClientCode>>,
    pub doc: Option<Vec<AttachmentDoc>>,
    pub code_file_list: Option<AttachmentCodeFileList>,
}

#[derive(Serialize, Deserialize)]
pub struct AttachmentCodeFileList {
    pub file_list: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)] // Mark the serde serialization format as untagged for backward compatibility: https://serde.rs/enum-representations.html#untagged
pub enum AttachmentDoc {
    Web(AttachmentWebDoc),
    Issue(AttachmentIssueDoc),
    Pull(AttachmentPullDoc),
}

#[derive(Serialize, Deserialize)]
pub struct AttachmentWebDoc {
    pub title: String,
    pub link: String,
    pub content: String,
}

#[derive(Serialize, Deserialize)]
pub struct AttachmentIssueDoc {
    pub title: String,
    pub link: String,
    pub author_user_id: Option<String>,
    pub body: String,
    pub closed: bool,
}

#[derive(Serialize, Deserialize)]
pub struct AttachmentPullDoc {
    pub title: String,
    pub link: String,
    pub author_user_id: Option<String>,
    pub body: String,
    pub diff: String,
    pub merged: bool,
}

#[derive(Serialize, Deserialize)]
pub struct AttachmentCode {
    pub git_url: String,
    pub commit: Option<String>,
    pub language: String,
    pub filepath: String,
    pub content: String,

    /// When start line is `None`, it represents the entire file.
    pub start_line: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct AttachmentClientCode {
    pub filepath: Option<String>,
    pub start_line: Option<usize>,
    pub content: String,
}
