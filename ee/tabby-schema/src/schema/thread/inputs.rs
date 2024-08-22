use juniper::{GraphQLInputObject, ID};
use tabby_common::api::code::CodeSearchParams;
use validator::Validate;

#[derive(GraphQLInputObject)]
pub struct CreateMessageInput {
    pub content: String,

    pub attachments: Option<MessageAttachmentInput>,
}

#[derive(GraphQLInputObject)]
pub struct CreateThreadInput {
    /// If true, the thread will be garbage collected after a certain amount of time.
    ///
    /// This is useful for threads that are not meant to be long-lived, such as those created from chat sidebars.
    #[graphql(default)]
    pub is_ephemeral: bool,

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

impl CodeSearchParamsOverrideInput {
    pub fn override_params(&self, params: &mut CodeSearchParams) {
        if let Some(min_embedding_score) = self.min_embedding_score {
            params.min_embedding_score = min_embedding_score as f32;
        }
        if let Some(min_bm25_score) = self.min_bm25_score {
            params.min_bm25_score = min_bm25_score as f32;
        }
        if let Some(min_rrf_score) = self.min_rrf_score {
            params.min_rrf_score = min_rrf_score as f32;
        }
        if let Some(num_to_return) = self.num_to_return {
            params.num_to_return = num_to_return as usize;
        }
        if let Some(num_to_score) = self.num_to_score {
            params.num_to_score = num_to_score as usize;
        }
    }
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

    #[graphql(default)]
    pub debug_options: Option<ThreadRunDebugOptionsInput>,
}

#[derive(GraphQLInputObject, Clone)]
pub struct CodeSearchParamsOverrideInput {
    pub min_embedding_score: Option<f64>,
    pub min_bm25_score: Option<f64>,
    pub min_rrf_score: Option<f64>,
    pub num_to_return: Option<i32>,
    pub num_to_score: Option<i32>,
}

#[derive(GraphQLInputObject, Clone)]
pub struct ThreadRunDebugOptionsInput {
    #[graphql(default)]
    pub code_search_params_override: Option<CodeSearchParamsOverrideInput>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadRunInput {
    pub thread_id: ID,

    pub additional_user_message: CreateMessageInput,

    #[graphql(default)]
    pub options: ThreadRunOptionsInput,
}

#[derive(GraphQLInputObject, Clone)]
pub struct MessageAttachmentInput {
    pub code: Vec<MessageAttachmentCodeInput>,
}

#[derive(GraphQLInputObject, Clone)]
pub struct MessageAttachmentCodeInput {
    pub filepath: Option<String>,
    pub start_line: Option<i32>,
    pub content: String,
}
