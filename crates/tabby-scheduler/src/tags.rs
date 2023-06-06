use std::fs::read_to_string;

use tabby_common::config::{Repository, Config};
use tree_sitter_tags::{TagsContext, TagsConfiguration};
use walkdir::{WalkDir, DirEntry};

pub fn tag_repositories(config: &Config) {
    let mut context = TagsContext::new();
    let js_config = TagsConfiguration::new(
        tree_sitter_javascript::language(),
        tree_sitter_javascript::TAGGING_QUERY,
        tree_sitter_javascript::LOCALS_QUERY
    ).unwrap();

    for repository in config.repositories.as_slice() {
        let walk_dir = WalkDir::new(repository.dir().as_path())
            .into_iter()
            .filter_entry(is_not_hidden)
            .filter_map(Result::ok)
            .filter(|e| !e.file_type().is_dir());

        for entry in walk_dir {
            if let Ok(source) = read_to_string(entry.path()) {
                let tags = context.generate_tags(&js_config, source.as_bytes(), None).unwrap().0.collect::<Result<Vec<_>, _>>().unwrap();
				for tag in tags {
                    println!("kind: {:?}", js_config.syntax_type_name(tag.syntax_type_id));
                    println!("name: {:?}", source.get(tag.name_range).unwrap());
				}
            }
        }
    }
}

fn is_not_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| entry.depth() == 0 || !s.starts_with('.'))
        .unwrap_or(false)
}
