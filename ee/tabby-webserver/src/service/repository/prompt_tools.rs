use std::sync::Arc;

use anyhow::Result;
use tabby_inference::ChatCompletionStream;
use tabby_schema::repository::{FileEntrySearchResult, Repository};

use crate::service::utils::prompt::{request_llm, transform_line_items};

// this is use to input a files tree, and generate some related question relate to repo dirs
pub async fn pipeline_related_questions_with_repo_dirs(
    chat: Arc<dyn ChatCompletionStream>,
    repository: &Repository,
    files: Vec<FileEntrySearchResult>,
    truncated: bool,
) -> Result<Vec<String>> {
    // Convert files into a formatted string for the prompt
    let mut files_content = files
        .iter()
        .map(|f| format!("Type: {}, Path: {}", f.r#type, f.path))
        .collect::<Vec<_>>()
        .join("\n");

    if truncated {
        files_content.push_str("\nNote: The file list has been truncated. There may be more files in subdirectories that were not included due to the limit.\n");
    }

    let mut prompt = format!(
        r#"You are a helpful assistant that helps the user to ask related questions about a codebase "{}".

Here is codebase directory structure:
{}"#,
        repository.name, files_content
    );

    prompt.push_str(
        r#"
Please generate 3 specific questions to help a new engineer understand this codebase.

Each question should be concise (max 10 words) and focused. Return only the questions, one per line.
"#,
    );

    let content = request_llm(chat, &prompt).await?;
    Ok(transform_line_items(&content))
}
