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
        let quota_threshold_for_snippets_from_code_search = 256;
        let mut max_snippets_chars_in_prompt = 768;
        let mut snippets: Vec<Snippet> = vec![];

        if let Some((count_characters, snippets_from_segments)) =
            extract_snippets_from_segments(max_snippets_chars_in_prompt, segments)
        {
            max_snippets_chars_in_prompt -= count_characters;
            snippets.extend(snippets_from_segments.into_iter());
        };

        if let Some(code) = &self.code {
            if max_snippets_chars_in_prompt > quota_threshold_for_snippets_from_code_search {
                let snippets_from_code_search = collect_snippets(
                    max_snippets_chars_in_prompt,
                    code.as_ref(),
                    language,
                    &segments.prefix,
                )
                .await;
                snippets.extend(snippets_from_code_search.into_iter());
            }
        }

        snippets
    }

    pub fn build(&self, language: &str, segments: Segments, snippets: &[Snippet]) -> String {
        let segments = rewrite_with_snippets(language, segments, snippets);
        self.build_prompt(segments.prefix, get_default_suffix(segments.suffix))
    }
}

fn get_default_suffix(suffix: Option<String>) -> String {
    suffix
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "\n".to_string())
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

fn extract_snippets_from_segments(
    max_snippets_chars: usize,
    segments: &Segments,
) -> Option<(usize, Vec<Snippet>)> {
    let mut count_characters = 0;
    let mut ret = Vec::new();

    // declarations has highest priority.
    if let Some(declarations) = &segments.declarations {
        for declaration in declarations {
            if count_characters + declaration.body.len() > max_snippets_chars {
                break;
            }

            count_characters += declaration.body.len();
            ret.push(Snippet {
                filepath: declaration.filepath.clone(),
                body: declaration.body.clone(),
                score: 1.0,
            });
        }
    }

    // then comes to the snippets from changed files.
    if let Some(relevant_snippets) = &segments.relevant_snippets_from_changed_files {
        for snippet in relevant_snippets {
            if count_characters + snippet.body.len() > max_snippets_chars {
                break;
            }

            count_characters += snippet.body.len();
            ret.push(Snippet {
                filepath: snippet.filepath.clone(),
                body: snippet.body.clone(),
                score: 1.0,
            });
        }
    }

    if ret.is_empty() {
        None
    } else {
        Some((count_characters, ret))
    }
}

async fn collect_snippets(
    max_snippets_chars: usize,
    code: &dyn CodeSearch,
    language: &str,
    text: &str,
) -> Vec<Snippet> {
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

        if count_characters + body.len() > max_snippets_chars {
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
    TOKENIZER
        .split(text)
        .map(|x| x.to_owned())
        .filter(|x| !x.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::completion::Declaration;

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
            filepath: None,
            git_url: None,
            declarations: None,
            relevant_snippets_from_changed_files: None,
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

    /// Empty strings tokens are not participating rag search and therefore could be removed.
    #[test]
    fn test_tokenized_text_filter() {
        let prefix = r#"public static String getFileExtension(String fullName) {
        String fileName = (new File(fullName)).getName();
        int dotIndex = fileName.lastIndexOf('.');
         }"#;

        // with filter
        assert_eq!(
            tokenize_text(prefix),
            [
                "public",
                "static",
                "String",
                "getFileExtension",
                "String",
                "fullName",
                "String",
                "fileName",
                "new",
                "File",
                "fullName",
                "getName",
                "int",
                "dotIndex",
                "fileName",
                "lastIndexOf",
            ]
        );
    }

    #[test]
    fn it_extract_snippets_from_segments() {
        let segments = Segments {
            prefix: "def fib(n):\n    ".to_string(),
            suffix: Some("\n        return fib(n - 1) + fib(n - 2)".to_string()),
            filepath: None,
            git_url: None,
            declarations: None,
            relevant_snippets_from_changed_files: None,
            clipboard: None,
        };

        let max_snippets_chars = 768;
        assert!(extract_snippets_from_segments(max_snippets_chars, &segments).is_none());

        let segments = Segments {
            prefix: "def fib(n):\n    ".to_string(),
            suffix: Some("\n        return fib(n - 1) + fib(n - 2)".to_string()),
            filepath: None,
            git_url: None,
            declarations: Some(vec![Declaration {
                filepath: "file:///path/to/file.py".to_string(),
                body: "def fib(n):\n    return n if n <= 1 else fib(n - 1) + fib(n - 2)"
                    .to_string(),
            }]),
            relevant_snippets_from_changed_files: Some(vec![Snippet {
                filepath: "a1.py".to_owned(),
                body: "res_1 = invoke_function_1(n)".to_owned(),
                score: 1.0,
            }]),
            clipboard: None,
        };

        assert!(
            extract_snippets_from_segments(max_snippets_chars, &segments)
                .is_some_and(|x| x.1.len() == 2)
        );
    }
}
