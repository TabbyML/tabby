use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_openai_alt::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use tabby_inference::ChatCompletionStream;
use tabby_schema::repository::FileEntrySearchResult;

/// Sends a prompt to the provided ChatCompletionStream and returns the generated response as a String.
pub async fn request_llm(chat: Arc<dyn ChatCompletionStream>, prompt: &str) -> Result<String> {
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
pub fn transform_line_items(content: &str) -> Vec<String> {
    content
        .lines()
        .map(trim_bullet)
        .filter(|x| !x.is_empty())
        .collect()
}

/// Trims leading and trailing bullet-like characters or digits from the provided string and returns the trimmed result.
pub fn trim_bullet(s: &str) -> String {
    let is_bullet = |c: char| c == '-' || c == '*' || c == '.' || c.is_numeric();
    s.trim()
        .trim_start_matches(is_bullet)
        .trim_end_matches(is_bullet)
        .trim()
        .to_owned()
}

/// Checks if the `check` string is contained within `content` in a case-insensitive manner.
pub fn detect_content(content: &str, check: &str) -> bool {
    content.to_lowercase().contains(check)
}

// this is use to input a files tree, and generate some related question relate to repo dirs
pub async fn pipeline_related_questions_with_repo_dirs(
    chat: Arc<dyn ChatCompletionStream>,
    files: Vec<FileEntrySearchResult>,
) -> Result<Vec<String>> {
    // Convert files into a formatted string for the prompt
    let files_content = files
        .iter()
        .map(|f| format!("Type: {}, Path: {}", f.r#type, f.path))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        r#"You are a helpful assistant that helps the user to ask related questions about a codebase structure.
Based on the following file structure, please generate 3 relevant questions that would help understand the codebase better.
Each question should be no longer than 20 words and be specific enough to stand alone.

File structure:
{}

Please generate 3 questions about this codebase structure that would help understand:
1. The organization and architecture
2. The main functionality
3. The potential implementation details

Return only the questions, one per line."#,
        files_content
    );

    let content = request_llm(chat, &prompt).await?;
    Ok(transform_line_items(&content))
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
