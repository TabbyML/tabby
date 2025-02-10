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

/// Sends a prompt to the provided ChatCompletionStream and returns the generated response as a String using a stream.
pub async fn request_llm_stream(
    chat: Arc<dyn ChatCompletionStream>,
    prompt: &str,
) -> BoxStream<'static, tabby_schema::Result<String>> {
    let prompt = prompt.to_owned();

    Box::pin(stream! {
        let request = CreateChatCompletionRequestArgs::default()
            .messages(vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt)
                    .build()
                    .expect("Failed to create ChatCompletionRequestUserMessage"),
            )])
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
