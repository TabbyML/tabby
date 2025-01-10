use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_openai_alt::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use tabby_inference::ChatCompletionStream;

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

fn detect_yes(content: &str) -> bool {
    content.to_lowercase().contains("yes")
}

/// Decide whether the question requires knowledge from codebase directory structure.
pub async fn pipeline_decide_need_codebase_directory_tree(
    chat: Arc<dyn ChatCompletionStream>,
    question: &str,
) -> Result<bool> {
    let prompt = format!(
        r#"You are a helpful assistant that helps the user to decide whether the question requires directory tree structure of the codebase. If it requires, return "Yes", otherwise return "No".

Here's a few examples:
"How many programming languages are used?" -> Yes
"Which module is used to compute 2D convolution?" -> Yes

Here's the original question:
{question}
"#
    );

    let content = request_llm(chat, &prompt).await?;
    Ok(detect_yes(&content))
}

/// Decide whether the question requires knowledge from codebase commit history.
pub async fn pipeline_decide_need_codebase_commit_history(
    chat: Arc<dyn ChatCompletionStream>,
    question: &str,
) -> Result<bool> {
    let prompt = format!(
        r#"You are a helpful assistant that helps the user to decide whether the question requires commit history of the codebase. If it requires, return "Yes", otherwise return "No".

Here's a few examples:
"What are recent changes to embedding api?" -> Yes

Here's the original question:
{question}
"#
    );

    let content = request_llm(chat, &prompt).await?;
    Ok(detect_yes(&content))
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

    #[test]
    fn test_detect_yes() {
        assert!(detect_yes("yes")); // Exact match
        assert!(detect_yes("YES")); // Uppercase
        assert!(detect_yes("Yes")); // Mixed case
        assert!(detect_yes("yess")); // Close but not exact
        assert!(detect_yes("This contains yes")); // Substring
        assert!(!detect_yes("no")); // Negative case
        assert!(!detect_yes("")); // Empty string
    }
}
