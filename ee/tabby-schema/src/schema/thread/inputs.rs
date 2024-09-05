use juniper::{GraphQLInputObject, ID};
use tabby_common::api::code::CodeSearchParams;
use validator::{Validate, ValidationError};

#[derive(GraphQLInputObject, Validate)]
pub struct CreateMessageInput {
    #[validate(length(
        min = 1,
        code = "content",
        message = "Message content should not be empty"
    ))]
    pub content: String,

    pub attachments: Option<MessageAttachmentInput>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadInput {
    #[validate(nested)]
    pub user_message: CreateMessageInput,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadAndRunInput {
    #[validate(nested)]
    pub thread: CreateThreadInput,

    #[graphql(default)]
    pub options: ThreadRunOptionsInput,
}

#[derive(GraphQLInputObject, Validate, Clone)]
pub struct DocQueryInput {
    pub content: String,

    /// Whether to collect documents from public web.
    pub search_public: bool,

    /// source_ids to be included in the doc search.
    pub source_ids: Option<Vec<String>>,
}

#[derive(GraphQLInputObject, Validate, Clone)]
#[validate(schema(function = "validate_code_query_input", skip_on_field_errors = false))]
pub struct CodeQueryInput {
    pub filepath: Option<String>,
    pub language: Option<String>,
    pub content: String,

    /// git_url to be included in the code search.
    pub git_url: Option<String>,

    /// source_ids to be included in the code search.
    pub source_id: Option<String>,
}

fn validate_code_query_input(input: &CodeQueryInput) -> Result<(), ValidationError> {
    if input.git_url.is_none() && input.source_id.is_none() {
        return Err(ValidationError::new("gitUrl")
            .with_message("Either gitUrl or sourceId must be provided".into()));
    }

    if input.git_url.is_some() && input.source_id.is_some() {
        return Err(ValidationError::new("gitUrl")
            .with_message("Only one of gitUrl or sourceId can be provided".into()));
    }

    Ok(())
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

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadRunInput {
    pub thread_id: ID,

    #[validate(nested)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_thread_run_input_shouldnot_allow_empty_content() {
        let input = CreateThreadRunInput {
            thread_id: ID::from("1".to_owned()),
            additional_user_message: CreateMessageInput {
                content: "".into(),
                attachments: None,
            },
            options: Default::default(),
        };

        assert!(input.validate().is_err())
    }
}
