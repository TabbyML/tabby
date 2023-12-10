use std::path::Path;
use anyhow::Result;
use tabby_common::Package;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct NpmPackage {
    name: String,
    version: String,
}

fn extract_npm_deps<'a, I>(packages: I) -> Vec<Package>
where
    I: IntoIterator<Item = &'a NpmPackage>,
{
    let mut res = packages
        .into_iter()
        .map(|npm_package| Package {
            language: String::from("javascript"),
            name: npm_package.name.clone(),
            version: Some(npm_package.version.clone())
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    res.sort_unstable();
    res
}

pub fn process_package_lock_json(path: &Path) -> Result<Vec<Package>> {
    let package_lock_file = path.join("package-lock.json");

    let file_content = std::fs::read_to_string(&package_lock_file)?;
    
    let npm_packages: Vec<NpmPackage> = serde_json::from_str(&file_content)?;
    
    let deps = extract_npm_deps(&npm_packages);
    Ok(deps)
}
