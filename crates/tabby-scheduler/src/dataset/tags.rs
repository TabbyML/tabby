use std::collections::HashMap;

use lazy_static::lazy_static;
use tabby_common::Tag;
use tree_sitter_tags::{TagsConfiguration, TagsContext};

pub fn collect(context: &mut TagsContext, language: &str, content: &str) -> Vec<Tag> {
    let config = LANGUAGE_TAGS.get(language);
    let empty = Vec::new();

    let Some(config) = config else {
        return empty;
    };

    let Ok((tags, has_error)) = context.generate_tags(&config.0, content.as_bytes(), None) else {
        return empty;
    };

    if has_error {
        return empty;
    }

    tags.filter_map(|x| x.ok())
        .map(|x| Tag {
            range: x.range,
            name_range: x.name_range,
            line_range: x.line_range,
            docs: x.docs,
            is_definition: x.is_definition,
            syntax_type_name: config.0.syntax_type_name(x.syntax_type_id).to_owned(),
        })
        .collect()
}

// Mark TagsConfiguration as thread sync / safe.
struct TagsConfigurationSync(TagsConfiguration);
unsafe impl Send for TagsConfigurationSync {}
unsafe impl Sync for TagsConfigurationSync {}

lazy_static! {
    static ref LANGUAGE_TAGS: HashMap<&'static str, TagsConfigurationSync> = {
        HashMap::from([
            (
                "python",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_python::language(),
                        tree_sitter_python::TAGGING_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "rust",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_rust::language(),
                        tree_sitter_rust::TAGGING_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "java",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_java::language(),
                        tree_sitter_java::TAGGING_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "kotlin",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_kotlin::language(),
                        include_str!("../../queries/kotlin.scm"),
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "javascript-typescript",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_typescript::language_tsx(),
                        include_str!("../../queries/tsx.scm"),
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "go",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_go::language(),
                        include_str!("../../queries/go.scm"),
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "ruby",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_ruby::language(),
                        tree_sitter_ruby::TAGGING_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "c",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_c::language(),
                        tree_sitter_c::TAGS_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "cpp",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_cpp::language(),
                        tree_sitter_cpp::TAGS_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
        ])
    };
}
