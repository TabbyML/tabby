use tabby_schema::page::PageSection;

pub fn prompt_page_title(title: Option<&str>) -> String {
    let prompt = if let Some(title) = title {
        format!(r#"Please help me to generate a page title for the input provided: {title}"#)
    } else {
        "Summarize the above conversation and create a succinct title that encapsulates its essence. Please only generate the title and nothing else. Do not include any additional text or context.".into()
    };

    format!("{prompt}\nPlease only generate the title and nothing else. Do not include any additional text.")
}

pub fn prompt_page_content(title: &str, page_section_titles: &[String]) -> String {
    let page_section_titles = page_section_titles.join("\n");
    format!(
        r#"
You're writing the intro section of a page named "{title}". It contains the following sub sections:
{page_section_titles}.

Please generate the content for the introduction section based on the information provided above.
Ensure the content is a single paragraph without any subtitles or nested sections.
Do not include any additional text.
"#
    )
}

fn generate_page_prompt(title: &str, sections: &[PageSection]) -> String {
    if sections.is_empty() {
        format!("You're writing a page named \"{}\".\n", title)
    } else {
        let sections = sections
            .iter()
            .map(|x| format!("## {}\n\n{}", x.title, x.content))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            r#"Here's some existing writings about the page.
=== Page Content Start ===
# {title}

{sections}
=== Page Content End ===
"#,
        )
    }
}

pub fn prompt_page_section_titles(
    count: usize,
    title: &str,
    sections: &[PageSection],
    new_section_prompt: Option<&str>,
) -> String {
    let page_prompt = generate_page_prompt(title, sections);

    let new_section = if let Some(new_section_prompt) = new_section_prompt {
        format!("The new section is about: {new_section_prompt}.\n")
    } else {
        "".to_string()
    };

    format!(
        r#"{page_prompt}{new_section}Please generate {count} section titles for the page based on above information.
Please only generate the section title and nothing else. Do not include any additional text.
Each section title should be on a new line.
"#
    )
}

pub fn prompt_page_section_content(
    title: &str,
    sections: &[PageSection],
    new_section_title: &str,
) -> String {
    let page_prompt = generate_page_prompt(title, sections);

    format!(
        r#"{page_prompt}.
The current new section title is: {new_section_title}

Please generate the content for the section based on the information provided above.
Ensure the content is a single paragraph without any subtitles or nested sections.
Do not include any additional text.
"#,
    )
}
