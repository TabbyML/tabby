pub fn prompt_page_title() -> &'static str {
    r#"
You are a helpful assistant that helps the user to write documents,
I am seeking your assistance in summarizing a conversation
and creating a succinct title that encapsulates its essence.
The title should not exceed 50 words,
and must be in the same language as the conversation provided.

To ensure the title accurately reflect the content,
please consider the context and key points discussed during the above dialogue.

Please do not include any additional information beyond the title itself,
and ensure no quotes or special characters or Title prefix are present in the title.

Please do not repeat the previous titles.

Please help me to generate a page title for the above conversation.
"#
}

pub fn prompt_page_content(title: &str) -> String {
    format!(
        r#"
You are a helpful assistant that helps the user to write documents,
I am requesting your support in crafting a page of a recent conversation I had.
you need to summary the conversation based on the conversation and context provided.
The summary should be concise, no more than 200 words.

The language used in the summary should correspond to that of the initial dialogue and existing content.
Your task is to distill the key points and main themes of the dialogue into a coherent and informative summary.

The title for the page is: {title}

Please refrain from duplicating the page title.
Please help me to generate a page title for the above conversation.
"#
    )
}

pub fn prompt_page_section_titles(count: usize, page: &str, new_section: &str) -> String {
    let new_section = if !new_section.is_empty() {
        format!("The new section is about: {new_section}.")
    } else {
        "".to_string()
    };

    format!(
        r#"
You are a helpful assistant that helps the user to write documents,
I am seeking your assistance in summarizing a conversation
and creating a succinct title that encapsulates its essence.
The title should not exceed 50 words,
and must be in the same language as the conversation provided.

To ensure the title accurately reflect the content,
please consider the context and key points discussed during the above dialogue.

Please do not include any additional information beyond the title itself,
and ensure no quotes or special characters or Title prefix are present in the title.

The current page title and content is:

```markdown
{page}
```

{new_section}

Please help me to generate {count} section titles for the above conversation and context,
please do not repeat the previous section titles.
"#
    )
}

pub fn prompt_page_section_content(page: &str, title: &str) -> String {
    format!(
        r#"
You are a helpful assistant that helps the user to write documents,
I am requesting your support in crafting a page of a recent conversation I had.
There may be some existed or not existed titles and contents in the page,
and you need to fill in the content based on the conversation and context provided.
The content should contain each key point and main theme of the dialogue base on the current title.

The language used in the content should correspond to that of the initial dialogue and existing content.
Your task is to provide a content base on the conversation, existing content, and current title.

There are some of the section titles and contents that have been generated:

```markdown
{page}
```

The current section title is: {title}

Please help me to generate this page section content using the above conversation as context,
Please make sure not to include the section title in the content.
"#,
    )
}
