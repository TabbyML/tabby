use tabby_schema::page::Section;

pub fn prompt_page_title(title: Option<&str>) -> String {
    let prompt = if let Some(title) = title {
        format!(r#"Please help me to generate a page title for the input provided: {title}"#)
    } else {
        "Summarize the above conversation and create a succinct title that encapsulates its essence. Please only generate the title and nothing else. Do not include any additional text or context.".into()
    };

    format!("{prompt}\nPlease only generate the title and nothing else. Do not include any additional text.")
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

fn generate_page_prompt(title: &str, sections: &[Section]) -> String {
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
}

pub fn prompt_page_section_titles(
    count: usize,
    title: &str,
    sections: &[Section],
    new_section_prompt: &str,
) -> String {
    let page_prompt = generate_page_prompt(title, sections);

    let new_section = if !new_section_prompt.is_empty() {
        format!("The new section is about: {new_section_prompt}.")
    } else {
        "".to_string()
    };

    format!(
        r#"{page_prompt}{new_section}

Please generate {count} section titles for the page based on above information.
Please only generate the section title and nothing else. Do not include any additional text.
Each section title should be on a new line.
"#
    )
}

pub fn prompt_page_section_content(
    title: &str,
    sections: &[Section],
    new_section_title: &str,
) -> String {
    let page_prompt = generate_page_prompt(title, sections);

    format!(
        r#"{page_prompt}.
The current new section title is: {new_section_title}

Please generate content of the section based on above information.
Please only generate the section title content nothing else. Do not include any additional text.
"#,
    )
}
