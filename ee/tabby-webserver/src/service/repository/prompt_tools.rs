use std::sync::Arc;

use anyhow::Result;
use tabby_inference::ChatCompletionStream;
use tabby_schema::repository::FileEntrySearchResult;

use crate::service::utils::prompt::{request_llm, transform_line_items};

// this is use to input a files tree, and generate some related question relate to repo dirs
pub async fn pipeline_related_questions_with_repo_dirs(
    chat: Arc<dyn ChatCompletionStream>,
    files: Vec<FileEntrySearchResult>,
    clipped: bool,
) -> Result<Vec<String>> {
    // Convert files into a formatted string for the prompt
    let files_content = files
        .iter()
        .map(|f| format!("Type: {}, Path: {}", f.r#type, f.path))
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt = format!(
        r#"You are a helpful assistant that helps the user to ask related questions about a codebase structure.
Based on the following file structure, please generate 3 relevant questions that would help understand the codebase better.
Each question should be no longer than 20 words and be specific enough to stand alone.

File structure:
{}"#,
        files_content
    );

    if clipped {
        prompt.push_str("\nNote: The file list has been clipped. There may be more files in subdirectories that were not included due to the limit.\n");
    }

    prompt.push_str(
        r#"
Please generate 3 questions about this codebase structure that would help understand:
1. The organization and architecture
2. The main functionality
3. The potential implementation details

Return only the questions, one per line."#,
    );

    let content = request_llm(chat, &prompt).await?;
    Ok(transform_line_items(&content))
}
