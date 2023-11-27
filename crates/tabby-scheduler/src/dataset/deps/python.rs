use std::{collections::HashSet, path::Path};

use anyhow::Result;
use tabby_common::Package;
use tracing::warn;

pub fn process_requirements_txt(path: &Path) -> Result<Vec<Package>> {
    let requirements_txt = path.join("requirements.txt");
    let content = std::fs::read_to_string(requirements_txt)?;

    let mut deps = HashSet::new();
    match requirements::parse_str(&content) {
        Ok(requirements) => {
            for requirement in requirements {
                if let Some(name) = requirement.name {
                    deps.insert(Package {
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

    Ok(deps.into_iter().collect())
}
