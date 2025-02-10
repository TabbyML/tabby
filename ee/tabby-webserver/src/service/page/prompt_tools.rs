use std::sync::Arc;

use anyhow::Result;
use tabby_inference::ChatCompletionStream;

use crate::service::utils::prompt::{request_llm, transform_line_items};

pub async fn pipeline_page_title(
    chat: Arc<dyn ChatCompletionStream>,
    context: &str,
    conversation: &str,
) -> Result<String> {
    let prompt = format!(
        r#"
You are a helpful assistant that helps the user to write documents,
I am seeking your assistance in summarizing a conversation
and creating a succinct title that encapsulates its essence.
The title should not exceed 50 words,
and must be in the same language as the conversation provided.

To ensure the summary and title accurately reflect the content,
please consider the context and key points discussed during our dialogue.

Please do not include any additional information beyond the title itself,
and ensure no quotes or special characters are present in the title.

Here are the contexts of the conversation:

{context}

Here is the conversation:

{conversation}
"#
    );

    request_llm(chat, &prompt).await
}

pub fn prompt_page_content(context: &str, conversation: &str) -> String {
    format!(
        r#"
You are a helpful assistant that helps the user to write documents,
I am requesting your support in crafting a summary of a recent conversation I had.
The summary should be concise, no more than 200 words, and presented in a single paragraph.
Please avoid using lists, code blocks, or any content that spans multiple lines.

The language of the summary should match that of the original conversation.
Your task is to distill the key points and main themes of the dialogue into a coherent and informative summary.

Here are the contexts of the conversation:

{context}

Here is the conversation:

{conversation}
"#
    )
}

pub async fn pipeline_page_sections(
    chat: Arc<dyn ChatCompletionStream>,
    context: &str,
    conversation: &str,
) -> Result<Vec<String>> {
    let prompt = format!(
        r#"
You are a helpful assistant that helps the user to write documents,
I require your expertise in distilling the essence of a conversation into exactly three distinct section titles.
Each title should succinctly reflect a major theme or topic from the dialogue,
and they must not exceed 50 words in length.

Please format the titles as bullet points using `-`, with each title on a separate line.

The language of the titles should be consistent with that of the original conversation.
No additional information is needed beyond the section titles themselves.

Here are the contexts of the conversation:

{context}

Here is the conversation:

{conversation}
"#
    );

    let content = request_llm(chat, &prompt).await?;
    Ok(transform_line_items(&content))
}

pub fn prompt_page_section_content(
    context: &str,
    conversation: &str,
    sections: &str,
    current: &str,
) -> String {
    format!(
        r#"
You are a helpful assistant that helps the user to write documents,
I need your help to create content for a specific section of a document I am working on.
The content should be based on a conversation I had,
taking into account the overall context and the relevant section titles.

For this task, please focus on expanding the themes and key points related to the current section title provided.
The section content should be no more than 200 words.

Please ensure that the language of the content corresponds with that of the original conversation.

Do not include the section title in the generated content.

Here are the contexts of the conversation:

{context}

Here is the conversation:

{conversation}

All the Section Titles:

{sections}

Current Section Title:

{current}
"#
    )
}
