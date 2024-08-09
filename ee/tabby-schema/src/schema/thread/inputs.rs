use juniper::{GraphQLInputObject, ID};
use validator::{Validate, ValidationError};

use super::Role;

#[derive(GraphQLInputObject, Validate)]
#[validate(schema(function = "validate_message_input", skip_on_field_errors = false))]
pub struct CreateMessageInput {
    role: Role,

    content: String,

    #[validate(nested)]
    attachments: Option<MessageAttachmentInput>,
}

#[derive(GraphQLInputObject, Validate)]
#[validate(schema(function = "validate_thread_input", skip_on_field_errors = false))]
pub struct CreateThreadInput {
    #[validate(nested)]
    messages: Vec<CreateMessageInput>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadAndRunInput {
    #[validate(nested)]
    pub thread: CreateThreadInput,

    #[validate(nested)]
    #[graphql(default)]
    pub options: ThreadRunOptionsInput,
}

#[derive(GraphQLInputObject, Validate)]
pub struct DocQueryInput {
    content: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CodeQueryInput {
    pub git_url: String,
    pub filepath: Option<String>,
    pub language: Option<String>,
    pub content: String,
}

#[derive(GraphQLInputObject, Validate, Default)]
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

    #[validate(nested)]
    pub additional_messages: Vec<CreateMessageInput>,

    #[validate(nested)]
    #[graphql(default)]
    pub options: ThreadRunOptionsInput,
}

#[derive(GraphQLInputObject, Validate)]
pub struct MessageAttachmentInput {
    #[validate(nested)]
    code: Vec<MessageAttachmentCodeInput>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct MessageAttachmentCodeInput {
    pub filepath: Option<String>,

    pub content: String,
}

fn validate_message_input(input: &CreateMessageInput) -> Result<(), ValidationError> {
    if let Role::Assistant = input.role {
        if input.attachments.is_some() {
            return Err(ValidationError::new(
                "Attachments are not allowed for assistants",
            ));
        }
    }

    Ok(())
}

fn validate_thread_input(input: &CreateThreadInput) -> Result<(), ValidationError> {
    let messages = &input.messages;
    let length = messages.len();

    for (i, message) in messages.iter().enumerate() {
        let is_last = i == length - 1;
        if !is_last && message.attachments.is_some() {
            return Err(ValidationError::new(
                "Attachments are only allowed on the last message",
            ));
        }
    }

    Ok(())
}
