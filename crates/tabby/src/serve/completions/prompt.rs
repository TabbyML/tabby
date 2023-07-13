use strfmt::strfmt;

use super::Segments;

pub struct PromptBuilder {
    prompt_template: Option<String>,
}

impl PromptBuilder {
    pub fn new(prompt_template: Option<String>) -> Self {
        PromptBuilder { prompt_template }
    }

    fn build_prompt(&self, prefix: String, suffix: String) -> String {
        if let Some(prompt_template) = &self.prompt_template {
            strfmt!(prompt_template, prefix => prefix, suffix => suffix).unwrap()
        } else {
            prefix
        }
    }

    pub fn build(&self, _language: &str, segments: Segments) -> String {
        let segments = self.rewrite(segments);
        if let Some(suffix) = segments.suffix {
            self.build_prompt(segments.prefix, suffix)
        } else {
            self.build_prompt(segments.prefix, "".to_owned())
        }
    }

    fn rewrite(&self, segments: Segments) -> Segments {
        segments
    }
}
