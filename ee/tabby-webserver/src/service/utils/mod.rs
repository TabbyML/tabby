pub mod prompt;

use std::sync::Arc;

use anyhow::{bail, Result};
use async_openai_alt::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, Role,
};
use tabby_schema::{
    context::{ContextInfoHelper, ContextService},
    policy::AccessPolicy,
    thread::{
        CodeQueryInput, Message, MessageAttachment, MessageAttachmentCodeInput,
        MessageAttachmentInput, Role as ThreadRole,
    },
};

pub async fn get_source_id(
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    input: &CodeQueryInput,
) -> Option<String> {
    let helper = context.read(Some(policy)).await.ok()?.helper();

    if let Some(source_id) = &input.source_id {
        if helper.can_access_source_id(source_id) {
            Some(source_id.clone())
        } else {
            None
        }
    } else if let Some(git_url) = &input.git_url {
        helper
            .allowed_code_repository()
            .closest_match(git_url)
            .map(|s| s.to_string())
    } else {
        None
    }
}

pub fn convert_messages_to_chat_completion_request(
    system_prompt: Option<&str>,
    helper: &ContextInfoHelper,
    messages: &[Message],
) -> Result<Vec<ChatCompletionRequestMessage>> {
    let mut output = vec![];
    output.reserve(messages.len() + 1);

    // System message
    if let Some(system_prompt) = system_prompt {
        output.push(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(system_prompt.to_owned()),
                name: None,
            },
        ));
    }

    for (i, x) in messages.iter().enumerate() {
        let role = match x.role {
            ThreadRole::Assistant => Role::Assistant,
            ThreadRole::User => Role::User,
        };

        let message: ChatCompletionRequestMessage = if role == Role::User {
            if i % 2 != 0 {
                bail!("User message must be followed by assistant message");
            }

            let y = &messages[i + 1];

            let user_attachment_input =
                user_attachment_input_from_user_message_attachment(&x.attachment);

            let content =
                build_user_prompt(&x.content, &y.attachment, Some(&user_attachment_input));
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(
                    helper.rewrite_tag(&content),
                ),
                ..Default::default()
            })
        } else {
            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                    x.content.clone(),
                )),
                ..Default::default()
            })
        };

        output.push(message);
    }

    Ok(output)
}

pub fn convert_user_message_to_chat_completion_request(
    helper: &ContextInfoHelper,
    query: &str,
    attachment: &MessageAttachment,
    user_attachment_input: Option<&MessageAttachmentInput>,
) -> ChatCompletionRequestMessage {
    let user_prompt = build_user_prompt(query, attachment, user_attachment_input);

    ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Text(helper.rewrite_tag(&user_prompt)),
        ..Default::default()
    })
}

fn user_attachment_input_from_user_message_attachment(
    attachment: &MessageAttachment,
) -> MessageAttachmentInput {
    let user_attachment_code_input: Vec<MessageAttachmentCodeInput> = attachment
        .client_code
        .iter()
        .map(Clone::clone)
        .map(Into::into)
        .collect();
    MessageAttachmentInput {
        code: user_attachment_code_input,
    }
}

pub fn build_user_prompt(
    user_input: &str,
    assistant_attachment: &MessageAttachment,
    user_attachment_input: Option<&MessageAttachmentInput>,
) -> String {
    // If the user message has no code attachment and the assistant message has no code attachment or doc attachment, return the user message directly.
    if user_attachment_input
        .map(|x| x.code.is_empty())
        .unwrap_or(true)
        && assistant_attachment.code.is_empty()
        && assistant_attachment.doc.is_empty()
        && assistant_attachment.code_file_list.is_none()
    {
        return user_input.to_owned();
    }

    let maybe_file_list_context = assistant_attachment
        .code_file_list
        .as_ref()
        .filter(|x| !x.file_list.is_empty())
        .map(|x| {
            format!(
                "Here is the list of files in the workspace available for reference:\n\n{}\n\n",
                x.file_list.join("\n")
            )
        })
        .unwrap_or_default();

    let maybe_file_snippet_context = {
        let snippets: Vec<String> = assistant_attachment
            .doc
            .iter()
            .map(|doc| format!("```\n{}\n```", doc.content()))
            .chain(
                user_attachment_input
                    .map(|x| &x.code)
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|snippet| {
                        if let Some(filepath) = &snippet.filepath {
                            format!("```title=\"{}\"\n{}\n```", filepath, snippet.content)
                        } else {
                            format!("```\n{}\n```", snippet.content)
                        }
                    }),
            )
            .chain(assistant_attachment.code.iter().map(|snippet| {
                format!(
                    "```{} title=\"{}\"\n{}\n```",
                    snippet.language, snippet.filepath, snippet.content
                )
            }))
            .collect();

        let citations: Vec<String> = snippets
            .iter()
            .enumerate()
            .map(|(i, snippet)| format!("[[citation:{}]]\n{}", i + 1, *snippet))
            .collect();

        if !citations.is_empty() {
            format!(
                "Here are the set of contexts:\n\n{}",
                citations.join("\n\n")
            )
        } else {
            String::default()
        }
    };

    format!(
        r#"You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [[citation:x]]. If a sentence comes from multiple contexts, please list all applicable citations, like [[citation:3]][[citation:5]]. Other than code and specific names and citations, your answer must be written in the same language as the question.

{maybe_file_list_context}{maybe_file_snippet_context}

Remember, don't blindly repeat the contexts verbatim. When possible, give code snippet to demonstrate the answer. And here is the user question:

{user_input}
"#
    )
}
