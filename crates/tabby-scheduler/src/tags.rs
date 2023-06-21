use anyhow::Result;
use tabby_common::{config::Config, Document};
use tree_sitter_tags::{TagsConfiguration, TagsContext};

pub fn tag_repositories(_config: &Config) -> Result<()> {
    let mut context = TagsContext::new();
    let js_config = TagsConfiguration::new(
        tree_sitter_javascript::language(),
        tree_sitter_javascript::TAGGING_QUERY,
        tree_sitter_javascript::LOCALS_QUERY,
    )
    .unwrap();

    for doc in Document::all()? {
        if doc.language == "javascript" {
            let tags = context
                .generate_tags(&js_config, doc.content.as_bytes(), None)?
                .0
                .collect::<Result<Vec<_>, _>>()?;
            for tag in tags {
                println!("kind: {:?}", js_config.syntax_type_name(tag.syntax_type_id));
                println!("name: {:?}", doc.content.get(tag.name_range).unwrap());
            }
        }
    }

    Ok(())
}
