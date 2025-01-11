use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_openai_alt::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use tabby_inference::ChatCompletionStream;
use tracing::debug;

async fn request_llm(chat: Arc<dyn ChatCompletionStream>, prompt: &str) -> Result<String> {
    let request = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()
                .expect("Failed to create ChatCompletionRequestUserMessage"),
        )])
        .build()?;

    let s = chat.chat(request).await?;
    let content = s.choices[0]
        .message
        .content
        .as_deref()
        .ok_or_else(|| anyhow!("Failed to get content from chat completion"))?;

    Ok(content.into())
}

/// Extracts items from the given content.
/// Assumptions:
/// 1. Each item is on a new line.
/// 2. Items may start with a bullet point, which needs to be trimmed.
fn transform_line_items(content: &str) -> Vec<String> {
    content
        .lines()
        .map(trim_bullet)
        .filter(|x| !x.is_empty())
        .collect()
}

fn trim_bullet(s: &str) -> String {
    let is_bullet = |c: char| c == '-' || c == '*' || c == '.' || c.is_numeric();
    s.trim()
        .trim_start_matches(is_bullet)
        .trim_end_matches(is_bullet)
        .trim()
        .to_owned()
}

/// Given context and a question, generate related questions.
pub async fn pipeline_related_questions(
    chat: Arc<dyn ChatCompletionStream>,
    context: &str,
    question: &str,
) -> Result<Vec<String>> {
    let prompt = format!(
        r#"
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:

{question}
"#
    );

    let content = request_llm(chat, &prompt).await?;
    Ok(transform_line_items(&content))
}

fn detect_content(content: &str, check: &str) -> bool {
    content.to_lowercase().contains(check)
}

#[derive(Debug)]
pub struct CodebaseContext {
    pub snippet: bool,
    pub file_list: bool,
}

/// Decide whether the question requires knowledge from codebase content.
pub async fn pipeline_decide_need_codebase_context(
    chat: Arc<dyn ChatCompletionStream>,
    question: &str,
) -> Result<CodebaseContext> {
    let prompt = format!(
        r#"You are a helpful assistant that helps the user to decide the types of context needed to answer the question. Currently, the following two kinds of context are supported:
SNIPPET: Snippets searched from codebase given the question.
FILE_LIST: File list of the codebase.

Your answer shall only contains raw string of context type, separated by comma.

Here's a few examples:
"How to implement an embedding api?" -> SNIPPET
"Which file contains http api definitions" -> SNIPPET,FILE_LIST
"How many python files is in the codebase?" -> FILE_LIST

Here's the original question:
{question}
"#
    );

    let content = request_llm(chat, &prompt).await?;
    let context = CodebaseContext {
        snippet: detect_content(&content, "snippet"),
        file_list: detect_content(&content, "file_list"),
    };
    debug!("decide_need_codebase_context: {:?} {:?}", content, context);
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_bullet() {
        assert_eq!(trim_bullet("- Hello"), "Hello");
        assert_eq!(trim_bullet("* World"), "World");
        assert_eq!(trim_bullet("1. Test"), "Test");
        assert_eq!(trim_bullet(".Dot"), "Dot");

        assert_eq!(trim_bullet("- Hello -"), "Hello");
        assert_eq!(trim_bullet("1. Test 1"), "Test");

        assert_eq!(trim_bullet("--** Mixed"), "Mixed");

        assert_eq!(trim_bullet("  - Hello  "), "Hello");

        assert_eq!(trim_bullet("-"), "");
        assert_eq!(trim_bullet(""), "");
        assert_eq!(trim_bullet("   "), "");

        assert_eq!(trim_bullet("Hello World"), "Hello World");

        assert_eq!(trim_bullet("1. *Bold* and -italic-"), "*Bold* and -italic");
    }
}
