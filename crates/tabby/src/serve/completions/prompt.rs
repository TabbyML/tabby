use std::sync::Arc;

use strfmt::strfmt;
use tracing::warn;

use super::Segments;
use crate::serve::{completions::languages::get_language, search::IndexServer};

static MAX_SNIPPETS_TO_FETCH: usize = 20;
static MAX_SNIPPET_CHARS_IN_PROMPT: usize = 512;
static SNIPPET_SCORE_THRESHOLD: f32 = 5.0;

pub struct PromptBuilder {
    prompt_template: Option<String>,
    index_server: Option<Arc<IndexServer>>,
}

impl PromptBuilder {
    pub fn new(prompt_template: Option<String>, index_server: Option<Arc<IndexServer>>) -> Self {
        PromptBuilder {
            prompt_template,
            index_server,
        }
    }

    fn build_prompt(&self, prefix: String, suffix: String) -> String {
        if let Some(prompt_template) = &self.prompt_template {
            strfmt!(prompt_template, prefix => prefix, suffix => suffix).unwrap()
        } else {
            prefix
        }
    }

    pub fn build(&self, language: &str, segments: Segments) -> String {
        let segments = self.rewrite(language, segments);
        self.build_prompt(segments.prefix, get_default_suffix(segments.suffix))
    }

    fn rewrite(&self, language: &str, segments: Segments) -> Segments {
        if let Some(index_server) = &self.index_server {
            rewrite_with_index(index_server, language, segments)
        } else {
            segments
        }
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

fn rewrite_with_index(
    index_server: &Arc<IndexServer>,
    language: &str,
    segments: Segments,
) -> Segments {
    let snippets = collect_snippets(index_server, language, &segments.prefix);
    if snippets.is_empty() {
        segments
    } else {
        let prefix = build_prefix(language, &segments.prefix, snippets, MAX_SNIPPET_CHARS_IN_PROMPT);
        Segments { prefix, ..segments }
    }
}

fn build_prefix(language: &str, prefix: &str, snippets: Vec<Snippet>, max_snippet_chars_in_prompt: usize) -> String {
    if snippets.is_empty() {
        return prefix.to_owned();
    }

    let comment_char = get_language(language).line_comment;
    let mut lines: Vec<String> = vec![];

    let mut count_characters = 0;
    for (i, snippet) in snippets.iter().enumerate() {
        let len = snippet.body.len();
        if count_characters + len > max_snippet_chars_in_prompt {
            break;
        }

        lines.push(format!("Path: {}", snippet.filepath));
        for line in snippet.body.lines() {
            lines.push(line.to_owned());
        }

        if i < snippets.len() - 1 {
            lines.push("".to_owned());
        }
        count_characters += len;
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

struct Snippet {
    filepath: String,
    body: String,
}

fn collect_snippets(index_server: &IndexServer, language: &str, text: &str) -> Vec<Snippet> {
    let mut ret = Vec::new();
    let sanitized_text = sanitize_text(text);
    if sanitized_text.is_empty() {
        return ret;
    }

    let query_text = format!("language:{} AND ({})", language, sanitized_text);

    let serp = match index_server.search(&query_text, MAX_SNIPPETS_TO_FETCH, 0) {
        Ok(serp) => serp,
        Err(err) => {
            warn!("Failed to search query: {}", err);
            return ret;
        }
    };

    for hit in serp.hits {
        if hit.score < SNIPPET_SCORE_THRESHOLD {
            break;
        }

        let body = hit.doc.body;

        if text.contains(&body) {
            // Exclude snippets already in the context window.
            continue;
        }

        ret.push(Snippet {
            filepath: hit.doc.filepath,
            body
        });
    }

    ret
}

fn sanitize_text(text: &str) -> String {
    // Only keep [a-zA-Z0-9-_]
    let x = text.replace(
        |c: char| !c.is_ascii_digit() && !c.is_alphabetic() && c != '_' && c != '-',
        " ",
    );
    let tokens: Vec<&str> = x
        .split(' ')
        .filter(|x| *x != "AND" && *x != "NOT" && *x != "OR" && x.len() > 5)
        .collect();
    tokens.join(" ")
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

    #[test]
    fn test_prompt_template() {
        let pb = create_prompt_builder(true);

        // Rewrite disabled, so the language doesn't matter.
        let language = "python";

        // Test w/ prefix, w/ suffix.
        {
            let segments = Segments {
                prefix: "this is some prefix".into(),
                suffix: Some("this is some suffix".into()),
            };
            assert_eq!(
                pb.build(language, segments),
                "<PRE> this is some prefix <SUF>this is some suffix <MID>"
            );
        }

        // Test w/ prefix, w/o suffix.
        {
            let segments = Segments {
                prefix: "this is some prefix".into(),
                suffix: None,
            };
            assert_eq!(
                pb.build(language, segments),
                "<PRE> this is some prefix <SUF>\n <MID>"
            );
        }

        // Test w/ prefix, w/ empty suffix.
        {
            let segments = Segments {
                prefix: "this is some prefix".into(),
                suffix: Some("".into()),
            };
            assert_eq!(
                pb.build(language, segments),
                "<PRE> this is some prefix <SUF>\n <MID>"
            );
        }

        // Test w/ empty prefix, w/ suffix.
        {
            let segments = Segments {
                prefix: "".into(),
                suffix: Some("this is some suffix".into()),
            };
            assert_eq!(
                pb.build(language, segments),
                "<PRE>  <SUF>this is some suffix <MID>"
            );
        }

        // Test w/ empty prefix, w/o suffix.
        {
            let segments = Segments {
                prefix: "".into(),
                suffix: None,
            };
            assert_eq!(pb.build(language, segments), "<PRE>  <SUF>\n <MID>");
        }

        // Test w/ emtpy prefix, w/ empty suffix.
        {
            let segments = Segments {
                prefix: "".into(),
                suffix: Some("".into()),
            };
            assert_eq!(pb.build(language, segments), "<PRE>  <SUF>\n <MID>");
        }
    }

    #[test]
    fn test_no_prompt_template() {
        let pb = create_prompt_builder(false);

        // Rewrite disabled, so the language doesn't matter.
        let language = "python";

        // Test w/ prefix, w/ suffix.
        {
            let segments = Segments {
                prefix: "this is some prefix".into(),
                suffix: Some("this is some suffix".into()),
            };
            assert_eq!(pb.build(language, segments), "this is some prefix");
        }

        // Test w/ prefix, w/o suffix.
        {
            let segments = Segments {
                prefix: "this is some prefix".into(),
                suffix: None,
            };
            assert_eq!(pb.build(language, segments), "this is some prefix");
        }

        // Test w/ prefix, w/ empty suffix.
        {
            let segments = Segments {
                prefix: "this is some prefix".into(),
                suffix: Some("".into()),
            };
            assert_eq!(pb.build(language, segments), "this is some prefix");
        }

        // Test w/ empty prefix, w/ suffix.
        {
            let segments = Segments {
                prefix: "".into(),
                suffix: Some("this is some suffix".into()),
            };
            assert_eq!(pb.build(language, segments), "");
        }

        // Test w/ empty prefix, w/o suffix.
        {
            let segments = Segments {
                prefix: "".into(),
                suffix: None,
            };
            assert_eq!(pb.build(language, segments), "");
        }

        // Test w/ empty prefix, w/ empty suffix.
        {
            let segments = Segments {
                prefix: "".into(),
                suffix: Some("".into()),
            };
            assert_eq!(pb.build(language, segments), "");
        }
    }

    #[test]
    fn test_build_prefix_readable() {
        let snippets = vec![
            Snippet {
                filepath: "a1.py".to_owned(),
                body: "res_1 = invoke_function_1(n)".to_owned(),
            },
            Snippet {
                filepath: "a2.py".to_owned(),
                body: "res_2 = invoke_function_2(n)".to_owned(),
            },
            Snippet {
                filepath: "a3.py".to_owned(),
                body: "res_3 = invoke_function_3(n)".to_owned(),
            },
            Snippet {
                filepath: "a4.py".to_owned(),
                body: "res_4 = invoke_function_4(n)".to_owned(),
            },
            Snippet {
                filepath: "a5.py".to_owned(),
                body: "res_5 = invoke_function_5(n)".to_owned(),
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
            build_prefix("python", prefix, snippets, 96),
            expected_built_prefix
        );
    }
}
