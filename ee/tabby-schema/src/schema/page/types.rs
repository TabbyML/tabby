use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject, GraphQLUnion, ID};
use validator::Validate;

use crate::{
    juniper::relay::NodeType,
    retrieval::{
        AttachmentCode, AttachmentCodeFileList, AttachmentCodeHit, AttachmentDoc, AttachmentDocHit,
    },
    thread::{CodeQueryInput, DocQueryInput, MessageAttachment},
    ChatCompletionMessage, Context,
};

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct Page {
    pub id: ID,
    pub author_id: ID,
    pub title: Option<String>,
    pub code_source_id: Option<String>,
    pub content: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl NodeType for Page {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "PageConnection"
    }

    fn edge_type_name() -> &'static str {
        "PageEdge"
    }
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct PageSection {
    pub id: ID,
    pub page_id: ID,
    pub title: String,
    pub content: String,
    pub position: i32,

    pub attachments: SectionAttachment,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    pub debug_data: Option<PageSectionDebugData>,
}

impl NodeType for PageSection {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "SectionConnection"
    }

    fn edge_type_name() -> &'static str {
        "SectionEdge"
    }
}

#[derive(GraphQLInputObject, Default)]
pub struct ThreadToPageDebugOptionInput {
    #[graphql(default)]
    pub return_chat_completion_request: bool,

    #[graphql(default)]
    pub return_query_request: bool,
}

#[derive(GraphQLInputObject)]
pub struct CreateThreadToPageRunInput {
    pub thread_id: ID,

    #[graphql(default)]
    pub debug_option: Option<ThreadToPageDebugOptionInput>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdatePageTitleInput {
    pub id: ID,
    #[validate(length(min = 1, max = 256, code = "title", message = "title can not be empty"))]
    pub title: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdatePageContentInput {
    pub id: ID,
    #[validate(length(
        min = 1,
        max = 65535,
        code = "content",
        message = "content can not be empty"
    ))]
    pub content: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdatePageSectionTitleInput {
    pub id: ID,
    #[validate(length(min = 1, max = 256, code = "title", message = "title can not be empty"))]
    pub title: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdatePageSectionContentInput {
    pub id: ID,
    #[validate(length(
        min = 1,
        max = 65535,
        code = "content",
        message = "content can not be empty"
    ))]
    pub content: String,
}

#[derive(GraphQLInputObject, Default)]
pub struct PageRunDebugOptionInput {
    #[graphql(default)]
    pub return_chat_completion_request: bool,

    #[graphql(default)]
    pub return_query_request: bool,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreatePageRunInput {
    pub title_prompt: String,

    #[validate(nested)]
    #[graphql(default)]
    pub doc_query: Option<DocQueryInput>,

    #[validate(nested)]
    #[graphql(default)]
    pub code_query: Option<CodeQueryInput>,

    #[graphql(default)]
    pub debug_option: Option<PageRunDebugOptionInput>,
}

#[derive(GraphQLInputObject, Default, Clone)]
pub struct PageSectionRunDebugOptionInput {
    #[graphql(default)]
    pub return_chat_completion_request: bool,

    #[graphql(default)]
    pub return_query_request: bool,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreatePageSectionRunInput {
    pub page_id: ID,
    pub title_prompt: String,

    #[validate(nested)]
    #[graphql(default)]
    pub doc_query: Option<DocQueryInput>,

    #[graphql(default)]
    pub debug_option: Option<PageSectionRunDebugOptionInput>,
}

#[derive(GraphQLEnum)]
pub enum MoveSectionDirection {
    Up,
    Down,
}

#[derive(GraphQLObject, Clone)]
pub struct PageTitleDebugData {
    /// Messages sent to LLM to generate the page title.
    pub generate_page_title_messages: Vec<ChatCompletionMessage>,
}

#[derive(GraphQLObject)]
pub struct PageCreated {
    pub id: ID,
    pub author_id: ID,
    pub title: String,

    pub debug_data: Option<PageTitleDebugData>,
}

#[derive(GraphQLObject, Clone)]
pub struct PageSectionDebugData {
    /// Messages sent to LLM to generate the page section titles.
    pub generate_section_titles_messages: Vec<ChatCompletionMessage>,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct PageSectionCreated {
    pub id: ID,
    pub page_id: ID,
    pub title: String,
    pub position: i32,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    pub debug_data: Option<PageSectionDebugData>,
}

impl From<PageSection> for PageSectionCreated {
    fn from(value: PageSection) -> Self {
        Self {
            id: value.id,
            page_id: value.page_id,
            title: value.title,
            position: value.position,
            created_at: value.created_at,
            updated_at: value.updated_at,
            debug_data: None,
        }
    }
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct PageSectionsCreated {
    pub sections: Vec<PageSectionCreated>,

    pub debug_data: Option<PageSectionDebugData>,
}

#[derive(GraphQLObject)]
pub struct PageSectionAttachmentCodeFileList {
    pub id: ID,
    pub code_file_list: AttachmentCodeFileList,
}

#[derive(GraphQLObject)]
pub struct PageSectionAttachmentCode {
    pub id: ID,
    pub codes: Vec<AttachmentCodeHit>,

    pub debug_data: Option<AttachmentCodeQueryDebugData>,
}

#[derive(GraphQLObject)]
pub struct AttachmentCodeQueryDebugData {
    pub source_id: String,
    pub query: String,
}

#[derive(GraphQLObject)]
pub struct AttachmentDocQueryDebugData {
    pub source_ids: Vec<String>,
    pub query: String,
}
#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct PageSectionAttachmentDoc {
    pub id: ID,
    pub doc: Vec<AttachmentDocHit>,

    pub debug_data: Option<AttachmentDocQueryDebugData>,
}

#[derive(GraphQLObject, Clone, Default)]
#[graphql(context = Context)]
pub struct SectionAttachment {
    pub code: Vec<AttachmentCode>,

    // FIXME(meng): consider remove code file list from section attachment.
    pub code_file_list: Option<AttachmentCodeFileList>,

    pub doc: Vec<AttachmentDoc>,
}

impl From<SectionAttachment> for MessageAttachment {
    fn from(attachment: SectionAttachment) -> MessageAttachment {
        MessageAttachment {
            client_code: vec![],
            code: attachment.code.iter().map(Into::into).collect(),
            code_file_list: attachment.code_file_list.map(Into::into),
            doc: vec![],
        }
    }
}

impl SectionAttachment {
    pub fn from_message_attachment(attachment: &MessageAttachment) -> SectionAttachment {
        SectionAttachment {
            code: attachment.code.iter().map(Into::into).collect(),
            code_file_list: attachment.code_file_list.as_ref().map(Into::into),
            doc: attachment.doc.iter().map(Into::into).collect(),
        }
    }

    pub fn merge(&mut self, other: &SectionAttachment) {
        for code in &other.code {
            if !self.code.iter().any(|c| c != code) {
                self.code.push(code.clone());
            }
        }

        if let Some(code_file_list) = &other.code_file_list {
            if self.code_file_list.is_none() {
                self.code_file_list = Some(code_file_list.clone());
            }
        }

        for doc in &other.doc {
            if !self.doc.iter().any(|d| d != doc) {
                self.doc.push(doc.clone());
            }
        }
    }
}

#[derive(GraphQLObject)]
pub struct PageContentDelta {
    pub delta: String,
}

#[derive(GraphQLObject, Clone)]
pub struct PageContentDebugData {
    /// Messages sent to LLM to generate the response.
    pub generate_page_content_messages: Vec<ChatCompletionMessage>,
}

#[derive(GraphQLObject)]
pub struct PageContentCompleted {
    pub id: ID,

    pub debug_data: Option<PageContentDebugData>,
}

#[derive(GraphQLObject)]
pub struct PageSectionContentDelta {
    pub id: ID,
    pub delta: String,
}

#[derive(GraphQLObject, Clone)]
pub struct PageSectionContentDebugData {
    /// Messages sent to LLM to generate the response.
    pub generate_section_content_messages: Vec<ChatCompletionMessage>,
}

#[derive(GraphQLObject)]
pub struct PageSectionContentCompleted {
    pub id: ID,

    pub debug_data: Option<PageSectionContentDebugData>,
}

#[derive(GraphQLObject)]
pub struct PageCompleted {
    pub id: ID,
}

/// Schema of page convert stream.
#[derive(GraphQLUnion)]
#[graphql(context = Context)]
pub enum PageRunItem {
    // PageCreated will return at the beginning of the stream,
    // containing the page ID, author and title.
    PageCreated(PageCreated),

    PageContentDelta(PageContentDelta),
    PageContentCompleted(PageContentCompleted),

    // PageSectionsCreated will return the titles of all sections.
    PageSectionsCreated(PageSectionsCreated),

    PageSectionAttachmentCodeFileList(PageSectionAttachmentCodeFileList),
    PageSectionAttachmentCode(PageSectionAttachmentCode),
    PageSectionAttachmentDoc(PageSectionAttachmentDoc),

    PageSectionContentDelta(PageSectionContentDelta),
    PageSectionContentCompleted(PageSectionContentCompleted),

    PageCompleted(PageCompleted),
}

/// Schema of page convert stream.
#[derive(GraphQLUnion)]
#[graphql(context = Context)]
pub enum SectionRunItem {
    PageSectionCreated(PageSectionCreated),

    PageSectionAttachmentCodeFileList(PageSectionAttachmentCodeFileList),
    PageSectionAttachmentCode(PageSectionAttachmentCode),
    PageSectionAttachmentDoc(PageSectionAttachmentDoc),

    PageSectionContentDelta(PageSectionContentDelta),
    PageSectionContentCompleted(PageSectionContentCompleted),
}
