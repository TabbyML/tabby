use std::collections::HashMap;

use lazy_static::lazy_static;
use tree_sitter_tags::TagsConfiguration;

// Mark TagsConfiguration as thread sync / safe.
pub(crate) struct TagsConfigurationSync(pub TagsConfiguration);
unsafe impl Send for TagsConfigurationSync {}
unsafe impl Sync for TagsConfigurationSync {}

pub fn get(language: &str) -> Option<&TagsConfigurationSync> {
    LANGUAGE_TAGS.get(language)
}

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
                        include_str!("../../queries/rust.scm"),
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
            (
                "csharp",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_c_sharp::language(),
                        include_str!("../../queries/csharp.scm"),
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "solidity",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_solidity::language(),
                        include_str!("../../queries/solidity.scm"),
                        "",
                    )
                    .unwrap(),
                ),
            ),
        ])
    };
}
