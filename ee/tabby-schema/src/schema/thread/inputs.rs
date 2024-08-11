use juniper::{GraphQLInputObject, ID};
use validator::{Validate, ValidationError};

use super::Role;

#[derive(GraphQLInputObject, Validate)]
#[validate(schema(function = "validate_message_input", skip_on_field_errors = false))]
pub struct CreateMessageInput {
    pub role: Role,

    pub content: String,

    #[validate(nested)]
    pub attachments: Option<MessageAttachmentInput>,
}

#[derive(GraphQLInputObject, Validate)]
#[validate(schema(function = "validate_thread_input", skip_on_field_errors = false))]
pub struct CreateThreadInput {
    #[validate(nested)]
    pub messages: Vec<CreateMessageInput>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadAndRunInput {
    #[validate(nested)]
    pub thread: CreateThreadInput,

    #[validate(nested)]
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
#[validate(schema(function = "validate_thread_run_input", skip_on_field_errors = false))]
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
    pub code: Vec<MessageAttachmentCodeInput>,
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
    validate_input_messages(&input.messages)
}

fn validate_thread_run_input(input: &CreateThreadRunInput) -> Result<(), ValidationError> {
    validate_input_messages(&input.additional_messages)
}

fn validate_input_messages(messages: &[CreateMessageInput]) -> Result<(), ValidationError> {
    let length = messages.len();

    for (i, message) in messages.iter().enumerate() {
        let is_last = i == length - 1;
        if !is_last && message.attachments.is_some() {
            return Err(ValidationError::new(
                "Attachments are only allowed on the last message",
            ));
        }

        if is_last {
            if message.role != Role::User {
                return Err(ValidationError::new(
                    "The last message must be from the user",
                ));
            }
        }

        let is_first = i == 0;
        if !is_first {
            let prev = &messages[i - 1];
            if prev.role == message.role {
                return Err(ValidationError::new(
                    "Cannot send two messages in a row with the same role",
                ));
            }
        }
    }

    Ok(())
}
