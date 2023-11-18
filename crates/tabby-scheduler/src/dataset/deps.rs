use std::{
    collections::HashSet,
    path::Path,
};

use anyhow::Result;
use serde::Deserialize;
use tabby_common::{Dependency, DependencyFile};
use tracing::{info, warn};

pub fn collect(path: &Path, file: &mut DependencyFile) {
    if let Ok(mut deps) = process_requirements_txt(path) {
        info!("Analyzing python requirements.txt ...");
        file.deps.append(&mut deps);
    }

    if let Ok(mut deps) = process_lock_file(path, "poetry.lock", "python") {
        info!("Analyzing python poetry.lock ...");
        file.deps.append(&mut deps);
    }

    if let Ok(mut deps) = process_lock_file(path, "Cargo.lock", "rust") {
        info!("Analyzing rust Cargo.lock ...");
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

#[derive(Deserialize)]
struct LockFileDependency {
    name: String,
    version: String,
}

#[derive(Deserialize)]
struct LockFile {
    package: Vec<LockFileDependency>,
}

fn process_lock_file(path: &Path, filename: &str, language: &str) -> Result<Vec<Dependency>> {
    let poetry_lock = path.join(filename);
    let deps: LockFile = serdeconv::from_toml_file(poetry_lock)?;
    Ok(deps
        .package
        .into_iter()
        .map(|x| Dependency {
            language: language.to_string(),
            name: x.name,
            version: Some(x.version),
        })
        .collect())
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
