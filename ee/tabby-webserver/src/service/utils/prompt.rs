use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_openai_alt::{
    error::OpenAIError,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use async_stream::stream;
use futures::stream::BoxStream;
use tabby_inference::ChatCompletionStream;
use tracing::{error, warn};

/// Sends a prompt to the provided ChatCompletionStream and returns the generated response as a String.
pub async fn request_llm_stream(
    chat: Arc<dyn ChatCompletionStream>,
    messages: Vec<ChatCompletionRequestMessage>,
) -> BoxStream<'static, tabby_schema::Result<String>> {
    Box::pin(stream! {
        let request = CreateChatCompletionRequestArgs::default()
            .messages(messages)
            .build().map_err(|e| anyhow!("Failed to build chat completion request: {:?}", e))?;

        let s = match chat.chat_stream(request).await {
            Ok(s) => s,
            Err(err) => {
                warn!("Failed to create chat completion stream: {:?}", err);
                return;
            }
        };

        for await chunk in s {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    if let OpenAIError::StreamError(content) = &err {
                        if content == "Stream ended" {
                            break;
                        }
                    }
                    error!("Failed to get chat completion chunk: {:?}", err);
                    yield Err(anyhow!("Failed to get chat completion chunk: {:?}", err).into());
                    return;
                }
            };

            let content = chunk.choices.first().and_then(|x| x.delta.content.as_deref());
            if let Some(content) = content {
                yield Ok(content.to_owned());
            }
        }
    })
}

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

/// Sends a prompt to the provided ChatCompletionStream and returns the generated response as a String.
pub async fn request_llm_with_message(
    chat: Arc<dyn ChatCompletionStream>,
    messages: Vec<ChatCompletionRequestMessage>,
) -> Result<String> {
    let request = CreateChatCompletionRequestArgs::default()
        .messages(messages)
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
        .map(|x| x.trim_bullet())
        .filter(|x| !x.is_empty())
        .map(|x| x.to_owned())
        .collect()
}

pub trait TrimTitle {
    fn trim_bullet(&self) -> &str;
    fn trim_title(&self) -> &str;
}

impl TrimTitle for str {
    /// Trims leading and trailing bullet-like characters or digits from the provided string and returns the trimmed result.
    fn trim_bullet(&self) -> &str {
        let s = self.trim();
        if s.starts_with('-') || s.starts_with('+') || s.starts_with('*') {
            return s.trim_matches(&['-', '+', '*', ' '][..]);
        }

        // Check for numbered list
        if let Some(pos) = s.find(". ") {
            let prefix = &s[..pos];
            if prefix.chars().all(|c| c.is_numeric()) {
                return s[pos + 2..].trim_start().trim_bullet();
            }
        }

        s
    }

    fn trim_title(&self) -> &str {
        let title = self.lines().next().unwrap_or(self).trim();
        title
            .trim_matches(&['"', '#', ' ', '-', '*'][..])
            .trim_bullet()
    }
}

/// Checks if the `check` string is contained within `content` in a case-insensitive manner.
pub fn detect_content(content: &str, check: &str) -> bool {
    content.to_lowercase().contains(check)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_bullet() {
        assert_eq!("- Hello".trim_bullet(), "Hello");
        assert_eq!("* World".trim_bullet(), "World");
        assert_eq!("1. Test".trim_bullet(), "Test");

        assert_eq!("- Hello -".trim_bullet(), "Hello");
        assert_eq!("1. Test 1".trim_bullet(), "Test 1");
        assert_eq!("12. Test 1".trim_bullet(), "Test 1");
        assert_eq!("1. Test 1.".trim_bullet(), "Test 1.");
        assert_eq!("1 Test ".trim_bullet(), "1 Test");

        assert_eq!("--** Mixed".trim_bullet(), "Mixed");

        assert_eq!("  - Hello  ".trim_bullet(), "Hello");

        assert_eq!("-".trim_bullet(), "");
        assert_eq!("".trim_bullet(), "");
        assert_eq!("   ".trim_bullet(), "");

        assert_eq!("Hello World".trim_bullet(), "Hello World");

        assert_eq!("1. *Bold* and -italic-".trim_bullet(), "Bold* and -italic");
        assert_eq!("1. **Bold* and -italic-".trim_bullet(), "Bold* and -italic");
    }

    #[test]
    fn test_trim_title() {
        assert_eq!("# - Hello".trim_title(), "Hello");
        assert_eq!("# * World".trim_title(), "World");
        assert_eq!("# 1. Test".trim_title(), "Test");

        assert_eq!("## - Hello -".trim_title(), "Hello");
        assert_eq!("  # 1. Test 1".trim_title(), "Test 1");
        assert_eq!("12. Test 1".trim_title(), "Test 1");
        assert_eq!(r#""1. Test 1.""#.trim_title(), "Test 1.");
        assert_eq!("## **Test ".trim_title(), "Test");
    }
}
