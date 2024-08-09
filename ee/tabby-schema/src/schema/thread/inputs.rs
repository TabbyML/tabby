use juniper::{GraphQLInputObject};
use validator::{Validate, ValidateLength, ValidationError};

use super::{Role};

#[derive(GraphQLInputObject, Validate)]
#[validate(schema(function = "validate_message_input", skip_on_field_errors = false))]
pub struct CreateMessageInput {
    role: Role,

    #[validate(length(code = "content", min = 1, message = "Content must not be empty"))]
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
pub struct CreateThreadRunInput {
    #[validate(nested)]
    pub thread: CreateThreadInput,
}

#[derive(GraphQLInputObject, Validate)]
pub struct MessageAttachmentInput {
    #[validate(nested)]
    code: Vec<MessageAttachmentCodeInput>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct MessageAttachmentCodeInput {
    pub filepath: Option<String>,

    #[validate(length(code = "content", min = 1, message = "Content must not be empty"))]
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
