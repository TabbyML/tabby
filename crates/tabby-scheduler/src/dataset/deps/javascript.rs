use std::{collections::HashMap, path::Path};

use package_lock_json_parser::PackageLockJson;
use tabby_common::Package;

fn extract_deps(lock_file: &PackageLockJson) -> Vec<Package> {
    let packages = lock_file
        .packages
        .as_ref()
        .map_or_else(HashMap::new, |p| p.clone());

    packages
        .values()
        .map(|pkg| Package {
            name: pkg.name.clone().unwrap_or_default(),
            version: Some(pkg.version.clone()),
            language: String::from("javascript"),
        })
        .collect()
}

pub fn process_package_lock_json(path: &Path) -> Result<Vec<Package>, Box<dyn std::error::Error>> {
    let package_lock_json_text = std::fs::read_to_string(path.join("package-lock.json"))?;
    let lock_file: PackageLockJson = package_lock_json_parser::parse(package_lock_json_text)?;

    Ok(extract_deps(&lock_file))
}
