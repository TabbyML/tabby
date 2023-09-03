use std::collections::HashMap;

use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use strfmt::strfmt;
use tabby_common::path::index_dir;
use tantivy::{
    collector::TopDocs, query::QueryParser, schema::Field, Index, ReloadPolicy, Searcher,
};
use tracing::{info, warn};

use super::Segments;

static MAX_SNIPPETS_TO_FETCH: usize = 20;
static MAX_SNIPPET_PER_NAME: u32 = 1;
static MAX_SNIPPET_CHARS_IN_PROMPT: usize = 1024;

pub struct PromptBuilder {
    prompt_template: Option<String>,
    index: Option<IndexState>,
}

impl PromptBuilder {
    pub fn new(prompt_template: Option<String>, enable_prompt_rewrite: bool) -> Self {
        let index = if enable_prompt_rewrite {
            info!("Experimental feature `enable_prompt_rewrite` is enabled, loading index ...");
            let index = IndexState::new();
            if let Err(err) = &index {
                warn!("Failed to open index in {:?}: {:?}", index_dir(), err);
            }
            index.ok()
        } else {
            None
        };

        PromptBuilder {
            prompt_template,
            index,
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
        if let Some(suffix) = segments.suffix {
            self.build_prompt(segments.prefix, suffix)
        } else {
            self.build_prompt(segments.prefix, "\n".to_owned())
        }
    }

    fn rewrite(&self, language: &str, segments: Segments) -> Segments {
        if let Some(index) = &self.index {
            rewrite_with_index(index, language, segments)
        } else {
            segments
        }
    }
}

fn rewrite_with_index(index: &IndexState, language: &str, segments: Segments) -> Segments {
    let snippets = collect_snippets(index, language, &segments.prefix);
    if snippets.is_empty() {
        segments
    } else {
        let prefix = build_prefix(language, &segments.prefix, snippets);
        Segments { prefix, ..segments }
    }
}

fn build_prefix(language: &str, prefix: &str, snippets: Vec<String>) -> String {
    let comment_char = LANGUAGE_LINE_COMMENT_CHAR.get(language).unwrap();
    let mut lines: Vec<String> = vec![format!(
        "Below are some relevant {} snippets found in the repository:",
        language
    )];

    let mut count_characters = 0;
    for (i, snippet) in snippets.iter().enumerate() {
        if count_characters + snippet.len() > MAX_SNIPPET_CHARS_IN_PROMPT {
            break;
        }

        lines.push(format!("== Snippet {} ==", i + 1));
        for line in snippet.lines() {
            lines.push(line.to_owned());
        }

        count_characters += snippet.len();
    }

    let commented_lines: Vec<String> = lines
        .iter()
        .map(|x| format!("{} {}", comment_char, x))
        .collect();
    let comments = commented_lines.join("\n");
    format!("{}\n{}", comments, prefix)
}

fn collect_snippets(index: &IndexState, language: &str, text: &str) -> Vec<String> {
    let mut ret = Vec::new();
    let sanitized_text = sanitize_text(text);
    if sanitized_text.is_empty() {
        return ret;
    }

    let query_text = format!(
        "language:{} AND kind:call AND ({})",
        language, sanitized_text
    );
    let query = match index.query_parser.parse_query(&query_text) {
        Ok(query) => query,
        Err(err) => {
            warn!("Failed to parse query: {}", err);
            return ret;
        }
    };

    let top_docs = index
        .searcher
        .search(&query, &TopDocs::with_limit(MAX_SNIPPETS_TO_FETCH))
        .unwrap();

    let mut names: HashMap<String, u32> = HashMap::new();
    for (_score, doc_address) in top_docs {
        let doc = index.searcher.doc(doc_address).unwrap();
        let name = doc
            .get_first(index.field_name)
            .and_then(|x| x.as_text())
            .unwrap();
        let count = *names.get(name).unwrap_or(&0);

        // Max 1 snippet per identifier.
        if count >= MAX_SNIPPET_PER_NAME {
            continue;
        }

        let body = doc
            .get_first(index.field_body)
            .and_then(|x| x.as_text())
            .unwrap();
        names.insert(name.to_owned(), count + 1);
        ret.push(body.to_owned());
    }

    ret
}

fn sanitize_text(text: &str) -> String {
    // Only keep [a-zA-Z0-9-_]
    let x = text.replace(
        |c: char| !c.is_ascii_digit() && !c.is_alphabetic() && c != '_' && c != '-',
        " ",
    );
    let tokens: Vec<&str> = x.split(' ').filter(|x| x.len() > 5).collect();
    tokens.join(" ")
}

struct IndexState {
    searcher: Searcher,
    query_parser: QueryParser,
    field_name: Field,
    field_body: Field,
}

impl IndexState {
    fn new() -> Result<IndexState> {
        let index = Index::open_in_dir(index_dir())?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()?;
        let field_name = index
            .schema()
            .get_field("name")
            .ok_or(anyhow!("Index doesn't have required field"))?;
        let field_body = index
            .schema()
            .get_field("body")
            .ok_or(anyhow!("Index doesn't have required field"))?;
        let query_parser = QueryParser::for_index(&index, vec![field_body]);
        Ok(Self {
            searcher: reader.searcher(),
            query_parser,
            field_name,
            field_body,
        })
    }
}

lazy_static! {
    static ref LANGUAGE_LINE_COMMENT_CHAR: HashMap<&'static str, &'static str> = HashMap::from([
        ("python", "#"),
        ("rust", "//"),
        ("javascript-typescript", "//"),
        ("go", "//"),
        ("java", "//"),
        ("lua", "--"),
    ]);
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
        PromptBuilder::new(prompt_template, false)
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
                "<PRE> this is some prefix <SUF> <MID>"
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
            assert_eq!(pb.build(language, segments), "<PRE>  <SUF> <MID>");
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
            "def snippet_1():\n    print(\"This is snipptet 1\")\n".to_string(),
            "def snippet_2():\n    print(\"This is snipptet 2\")\n".to_string(),
            "def snippet_3():\n    print(\"This is snipptet 3\")\n".to_string(),
            "def snippet_4():\n    print(\"This is snipptet 4\")\n".to_string(),
            "def snippet_5():\n    print(\"This is snipptet 5\")\n".to_string(),
        ];

        let prefix = "def this_is_prefix():\n";

        let expected_built_prefix = "\
# Below are some relevant python snippets found in the repository:
# == Snippet 1 ==
# def snippet_1():
#     print(\"This is snipptet 1\")
# == Snippet 2 ==
# def snippet_2():
#     print(\"This is snipptet 2\")
# == Snippet 3 ==
# def snippet_3():
#     print(\"This is snipptet 3\")
# == Snippet 4 ==\n# def snippet_4():
#     print(\"This is snipptet 4\")
# == Snippet 5 ==
# def snippet_5():
#     print(\"This is snipptet 5\")
def this_is_prefix():\n";

        assert_eq!(
            build_prefix("python", prefix, snippets),
            expected_built_prefix
        );
    }

    #[test]
    fn test_build_prefix_count_chars() {
        let snippets_expected = 4;
        let snippet_payload = "a".repeat(MAX_SNIPPET_CHARS_IN_PROMPT / snippets_expected);
        let mut snippets = vec![];
        for _ in 0..snippets_expected {
            snippets.push(snippet_payload.clone());
        }

        let prefix = "def this_is_prefix():\n";

        let generated_prompt = build_prefix("python", prefix, snippets);

        println!("{}", generated_prompt);

        for i in 0..snippets_expected + 1 {
            let st = format!("# == Snippet {} ==", i + 1);
            if i < snippets_expected {
                assert!(generated_prompt.contains(&st));
            } else {
                assert!(!generated_prompt.contains(&st));
            }
        }
    }
}
