use std::{error::Error, fs, path::Path};
use package_lock_json_parser::{parse_dependencies};
use tabby_common::Package;

pub fn process_js_package_lock(path: &Path) -> Result<Vec<Package>, Box<dyn Error>> {
    let package_lock_content = fs::read_to_string(path)?;
    let dependencies: Vec<Package> = parse_dependencies(package_lock_content)?
        .into_iter()
        .map(|dep| Package {
            language: String::from("javascript"),
            name: dep.name.to_string(),
            version: Some(dep.version.to_string()),
        })
        .collect();

    Ok(dependencies)
}