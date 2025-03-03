use tabby_schema::page::Section;

pub fn prompt_page_title(title: Option<&str>) -> String {
    let prompt = if let Some(title) = title {
        format!(r#"Please help me to generate a page title for the input provided: {title}"#)
    } else {
        "Summarize the above conversation and create a succinct title that encapsulates its essence. Please only generate the title and nothing else. Do not include any additional text or context.".into()
    };

    format!("{prompt}\nPlease only generate the title and nothing else. Do not include any additional text or context.")
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

pub fn prompt_page_section_titles(
    count: usize,
    title: &str,
    sections: &[Section],
    new_section_prompt: &str,
) -> String {
    let page_prompt = {
        let sections = sections
            .iter()
            .map(|x| format!("## {}\n\n{}", x.title, x.content))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            r#"Here's some existing writings about the page.
# {title}

{sections}
"#,
        )
    };

    let new_section = if !new_section_prompt.is_empty() {
        format!("The new section is about: {new_section_prompt}.")
    } else {
        "".to_string()
    };

    format!(
        r#"{page_prompt}{new_section}

Please generate {count} section titles for the page based on above information.
Please only generate the section title and nothing else. Do not include any additional text or context.
Each section title should be on a new line.
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
