use std::sync::Arc;

use anyhow::Result;
use tabby_inference::ChatCompletionStream;
use tabby_schema::thread::ThreadAssistantMessageReadingCode;
use tracing::debug;

use crate::service::utils::prompt::{detect_content, request_llm, transform_line_items};

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

/// Decide whether the question requires knowledge from codebase content.
pub async fn pipeline_decide_need_codebase_context(
    chat: Arc<dyn ChatCompletionStream>,
    question: &str,
) -> Result<ThreadAssistantMessageReadingCode> {
    let prompt = format!(
        r#"You are a helpful assistant that helps the user to decide the types of context needed to answer the question. Currently, the following three kinds of context are supported:
SNIPPET: Snippets searched from codebase given the question.
FILE_LIST: File list of the codebase.
COMMIT: Commit history of the codebase.

Your answer shall only contains raw string of context type, separated by comma.

Here's a few examples:
"How to implement an embedding api?" -> SNIPPET
"Which file contains http api definitions" -> SNIPPET,FILE_LIST
"How many python files is in the codebase?" -> FILE_LIST
"Which file contains main function?" -> SNIPPET
"Which file changed recently?" -> COMMIT
"When was the last commit?" -> COMMIT
"What is the most recent update to the embedding API?" -> COMMIT,SNIPPET

Here's the original question:
{question}
"#
    );

    let content = request_llm(chat, &prompt).await?;
    let context = ThreadAssistantMessageReadingCode {
        snippet: detect_content(&content, "snippet"),
        file_list: detect_content(&content, "file_list"),
        commit_history: detect_content(&content, "commit"),
    };
    debug!("decide_need_codebase_context: {:?} {:?}", content, context);
    Ok(context)
}
