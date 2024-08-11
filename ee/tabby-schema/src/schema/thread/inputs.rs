use juniper::{GraphQLInputObject, ID};
use validator::Validate;

#[derive(GraphQLInputObject)]
pub struct CreateMessageInput {
    pub content: String,

    pub attachments: Option<MessageAttachmentInput>,
}

#[derive(GraphQLInputObject)]
pub struct CreateThreadInput {
    pub user_message: CreateMessageInput,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadAndRunInput {
    pub thread: CreateThreadInput,

    #[graphql(default)]
    pub options: ThreadRunOptionsInput,
}

#[derive(GraphQLInputObject, Validate, Clone)]
pub struct DocQueryInput {
    pub content: String,
}

#[derive(GraphQLInputObject, Validate, Clone)]
pub struct CodeQueryInput {
    pub git_url: String,
    pub filepath: Option<String>,
    pub language: Option<String>,
    pub content: String,
}

#[derive(GraphQLInputObject, Validate, Default, Clone)]
pub struct ThreadRunOptionsInput {
    #[validate(nested)]
    #[graphql(default)]
    pub doc_query: Option<DocQueryInput>,

    #[validate(nested)]
    #[graphql(default)]
    pub code_query: Option<CodeQueryInput>,

    #[graphql(default)]
    pub generate_relevant_questions: bool,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadRunInput {
    pub thread_id: ID,

    pub additional_user_message: CreateMessageInput,

    #[graphql(default)]
    pub options: ThreadRunOptionsInput,
}

#[derive(GraphQLInputObject)]
pub struct MessageAttachmentInput {
    pub code: Vec<MessageAttachmentCodeInput>,
}

#[derive(GraphQLInputObject)]
pub struct MessageAttachmentCodeInput {
    pub filepath: Option<String>,

    pub content: String,
}
