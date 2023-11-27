use std::sync::Arc;

use lazy_static::lazy_static;
use regex::Regex;
use strfmt::strfmt;
use tabby_common::{
    api::code::{CodeSearch, CodeSearchError},
    languages::get_language,
};
use textdistance::Algorithm;
use tracing::warn;

use super::{Segments, Snippet};

static MAX_SNIPPETS_TO_FETCH: usize = 20;
static MAX_SNIPPET_CHARS_IN_PROMPT: usize = 768;
static MAX_SIMILARITY_THRESHOLD: f32 = 0.9;

pub struct PromptBuilder {
    prompt_template: Option<String>,
    code: Option<Arc<dyn CodeSearch>>,
}

impl PromptBuilder {
    pub fn new(prompt_template: Option<String>, code: Option<Arc<dyn CodeSearch>>) -> Self {
        PromptBuilder {
            prompt_template,
            code,
        }
    }

    fn build_prompt(&self, prefix: String, suffix: String) -> String {
        let Some(prompt_template) = &self.prompt_template else {
            return prefix;
        };

        strfmt!(prompt_template, prefix => prefix, suffix => suffix).unwrap()
    }

    pub async fn collect(&self, language: &str, segments: &Segments) -> Vec<Snippet> {
        if let Some(code) = &self.code {
            collect_snippets(code.as_ref(), language, &segments.prefix).await
        } else {
            vec![]
        }
    }

    pub fn build(&self, language: &str, segments: Segments, snippets: &[Snippet]) -> String {
        let segments = rewrite_with_snippets(language, segments, snippets);
        self.build_prompt(segments.prefix, get_default_suffix(segments.suffix))
    }
}

fn get_default_suffix(suffix: Option<String>) -> String {
    if suffix.is_none() {
        return "\n".to_owned();
    }

    let suffix = suffix.unwrap();
    if suffix.is_empty() {
        "\n".to_owned()
    } else {
        suffix
    }
}

fn rewrite_with_snippets(language: &str, segments: Segments, snippets: &[Snippet]) -> Segments {
    if snippets.is_empty() {
        segments
    } else {
        let prefix = build_prefix(language, &segments.prefix, snippets);
        Segments { prefix, ..segments }
    }
}

fn build_prefix(language: &str, prefix: &str, snippets: &[Snippet]) -> String {
    if snippets.is_empty() {
        return prefix.to_owned();
    }

    let comment_char = &get_language(language).line_comment;
    let mut lines: Vec<String> = vec![];

    for (i, snippet) in snippets.iter().enumerate() {
        lines.push(format!("Path: {}", snippet.filepath));
        for line in snippet.body.lines() {
            lines.push(line.to_owned());
        }

        if i < snippets.len() - 1 {
            lines.push("".to_owned());
        }
    }

    let commented_lines: Vec<String> = lines
        .iter()
        .map(|x| {
            if x.is_empty() {
                comment_char.to_string()
            } else {
                format!("{} {}", comment_char, x)
            }
        })
        .collect();
    let comments = commented_lines.join("\n");
    format!("{}\n{}", comments, prefix)
}

async fn collect_snippets(code: &dyn CodeSearch, language: &str, text: &str) -> Vec<Snippet> {
    let mut ret = Vec::new();
    let mut tokens = tokenize_text(text);

    let serp = match code
        .search_in_language(language, &tokens, MAX_SNIPPETS_TO_FETCH, 0)
        .await
    {
        Ok(serp) => serp,
        Err(CodeSearchError::NotReady) => {
            // Ignore.
            return vec![];
        }
        Err(CodeSearchError::TantivyError(err)) => {
            warn!("Failed to search: {}", err);
            return ret;
        }
        Err(CodeSearchError::QueryParserError(err)) => {
            warn!("Failed to parse query: {}", err);
            return ret;
        }
    };

    let mut count_characters = 0;
    for hit in serp.hits {
        let body = hit.doc.body;
        let mut body_tokens = tokenize_text(&body);

        if count_characters + body.len() > MAX_SNIPPET_CHARS_IN_PROMPT {
            break;
        }

        let similarity = if body_tokens.len() > tokens.len() {
            0.0
        } else {
            let distance = textdistance::LCSSeq::default()
                .for_iter(tokens.iter(), body_tokens.iter())
                .val() as f32;
            distance / body_tokens.len() as f32
        };

        if similarity > MAX_SIMILARITY_THRESHOLD {
            // Exclude snippets presents in context window.
            continue;
        }

        // Prepend body tokens and update tokens, so future similarity calculation will consider
        // added snippets.
        body_tokens.append(&mut tokens);
        tokens.append(&mut body_tokens);

        count_characters += body.len();
        ret.push(Snippet {
            filepath: hit.doc.filepath,
            body,
            score: hit.score,
        });
    }

    ret
}

lazy_static! {
    static ref TOKENIZER: Regex = Regex::new(r"[^\w]").unwrap();
}

fn tokenize_text(text: &str) -> Vec<String> {
    TOKENIZER.split(text).map(|x| x.to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_prompt_builder(with_template: bool) -> PromptBuilder {
        let prompt_template = if with_template {
            // Init prompt builder with codellama prompt template
            Some("<PRE> {prefix} <SUF>{suffix} <MID>".into())
        } else {
            None
        };

        // Init prompt builder with prompt rewrite disabled.
        PromptBuilder::new(prompt_template, None)
    }

    fn make_segment(prefix: String, suffix: Option<String>) -> Segments {
        Segments {
            prefix,
            suffix,
            clipboard: None,
        }
    }

    #[test]
    fn test_prompt_template() {
        let pb = create_prompt_builder(true);

        // Rewrite disabled, so the language doesn't matter.
        let language = "python";
        let snippets = &vec![];

        // Test w/ prefix, w/ suffix.
        {
            let segments = make_segment(
                "this is some prefix".into(),
                Some("this is some suffix".into()),
            );
            assert_eq!(
                pb.build(language, segments, snippets),
                "<PRE> this is some prefix <SUF>this is some suffix <MID>"
            );
        }

        // Test w/ prefix, w/o suffix.
        {
            let segments = make_segment("this is some prefix".into(), None);
            assert_eq!(
                pb.build(language, segments, snippets),
                "<PRE> this is some prefix <SUF>\n <MID>"
            );
        }

        // Test w/ prefix, w/ empty suffix.
        {
            let segments = make_segment("this is some prefix".into(), Some("".into()));
            assert_eq!(
                pb.build(language, segments, snippets),
                "<PRE> this is some prefix <SUF>\n <MID>"
            );
        }

        // Test w/ empty prefix, w/ suffix.
        {
            let segments = make_segment("".into(), Some("this is some suffix".into()));
            assert_eq!(
                pb.build(language, segments, snippets),
                "<PRE>  <SUF>this is some suffix <MID>"
            );
        }

        // Test w/ empty prefix, w/o suffix.
        {
            let segments = make_segment("".into(), None);
            assert_eq!(
                pb.build(language, segments, snippets),
                "<PRE>  <SUF>\n <MID>"
            );
        }

        // Test w/ emtpy prefix, w/ empty suffix.
        {
            let segments = make_segment("".into(), Some("".into()));
            assert_eq!(
                pb.build(language, segments, snippets),
                "<PRE>  <SUF>\n <MID>"
            );
        }
    }

    #[test]
    fn test_no_prompt_template() {
        let pb = create_prompt_builder(false);

        // Rewrite disabled, so the language doesn't matter.
        let language = "python";
        let snippets = &vec![];

        // Test w/ prefix, w/ suffix.
        {
            let segments = make_segment(
                "this is some prefix".into(),
                Some("this is some suffix".into()),
            );
            assert_eq!(
                pb.build(language, segments, snippets),
                "this is some prefix"
            );
        }

        // Test w/ prefix, w/o suffix.
        {
            let segments = make_segment("this is some prefix".into(), None);
            assert_eq!(
                pb.build(language, segments, snippets),
                "this is some prefix"
            );
        }

        // Test w/ prefix, w/ empty suffix.
        {
            let segments = make_segment("this is some prefix".into(), Some("".into()));
            assert_eq!(
                pb.build(language, segments, snippets),
                "this is some prefix"
            );
        }

        // Test w/ empty prefix, w/ suffix.
        {
            let segments = make_segment("".into(), Some("this is some suffix".into()));
            assert_eq!(pb.build(language, segments, snippets), "");
        }

        // Test w/ empty prefix, w/o suffix.
        {
            let segments = make_segment("".into(), None);
            assert_eq!(pb.build(language, segments, snippets), "");
        }

        // Test w/ empty prefix, w/ empty suffix.
        {
            let segments = make_segment("".into(), Some("".into()));
            assert_eq!(pb.build(language, segments, snippets), "");
        }
    }

    #[test]
    fn test_build_prefix_readable() {
        let snippets = vec![
            Snippet {
                filepath: "a1.py".to_owned(),
                body: "res_1 = invoke_function_1(n)".to_owned(),
                score: 1.0,
            },
            Snippet {
                filepath: "a2.py".to_owned(),
                body: "res_2 = invoke_function_2(n)".to_owned(),
                score: 1.0,
            },
            Snippet {
                filepath: "a3.py".to_owned(),
                body: "res_3 = invoke_function_3(n)".to_owned(),
                score: 1.0,
            },
        ];

        let prefix = "\
'''
Use some invoke_function to do some job.
'''
def this_is_prefix():\n";

        let expected_built_prefix = "\
# Path: a1.py
# res_1 = invoke_function_1(n)
#
# Path: a2.py
# res_2 = invoke_function_2(n)
#
# Path: a3.py
# res_3 = invoke_function_3(n)
'''
Use some invoke_function to do some job.
'''
def this_is_prefix():\n";

        assert_eq!(
            build_prefix("python", prefix, &snippets),
            expected_built_prefix
        );
    }
}
