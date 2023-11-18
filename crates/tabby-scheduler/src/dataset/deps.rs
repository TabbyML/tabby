use std::{collections::HashSet, path::Path};

use anyhow::Result;
use tabby_common::{Dependency, DependencyFile};
use tracing::warn;

pub fn collect(path: &Path, file: &mut DependencyFile) {
    if let Ok(mut deps) = process_requirements_txt(path) {
        file.deps.append(&mut deps);
    }

    remove_duplicates(file);
}

fn process_requirements_txt(path: &Path) -> Result<Vec<Dependency>> {
    let requirements_txt = path.join("requirements.txt");
    let content = std::fs::read_to_string(requirements_txt)?;

    let mut deps = vec![];
    match requirements::parse_str(&content) {
        Ok(requirements) => {
            for requirement in requirements {
                if let Some(name) = requirement.name {
                    deps.push(Dependency {
                        language: "python".to_owned(),
                        name,
                        version: None, // requirements.txt doesn't come with accurate version information.
                    });
                }
            }
        }
        Err(err) => {
            warn!("Failed to parse requirements.txt: {}", err);
        }
    }

    Ok(deps)
}

fn remove_duplicates(file: &mut DependencyFile) {
    let mut keys: HashSet<(String, String)> = HashSet::default();
    let mut deps = vec![];
    for x in &file.deps {
        let key = (x.language.clone(), x.name.clone());
        if !keys.contains(&key) {
            keys.insert(key);
            deps.push(x.clone());
        }
    }

    file.deps = deps;
}
