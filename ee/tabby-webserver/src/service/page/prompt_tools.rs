pub fn system_prompt_page_title() -> String {
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

Please do not repeat the previous titles.
"#
    .to_string()
}

pub fn system_prompt_page_content() -> String {
    r#"
You are a helpful assistant that helps the user to write documents,
I am requesting your support in crafting a page of a recent conversation I had.
you need to summary the conversation based on the conversation and context provided.
The summary should be concise, no more than 200 words.

The language used in the summary should correspond to that of the initial dialogue and existing content.
Your task is to distill the key points and main themes of the dialogue into a coherent and informative summary.
"#.to_string()
}

pub fn system_prompt_page_section_content() -> String {
    r#"
You are a helpful assistant that helps the user to write documents,
I am requesting your support in crafting a page of a recent conversation I had.
There may be some existed or not existed titles and contents in the page,
and you need to fill in the content based on the conversation and context provided.
The content should contain each key point and main theme of the dialogue base on the current title.

The language used in the content should correspond to that of the initial dialogue and existing content.
Your task is to provide a content base on the conversation, existing content, and current title.
"#.to_string()
}
