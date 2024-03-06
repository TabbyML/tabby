use std::{collections::HashSet, fs::read_to_string, path::Path};

use anyhow::Result;
use tabby_common::Package;

fn process_package_json(path: &Path) -> Result<Vec<Package>> {
    let package_json_file = path.join("package.json");
    let mut package_json_contents = npm_package_json::Package::from_path(package_json_file)?;
    let mut package_deps = package_json_contents.dependencies;
    package_deps.append(&mut package_json_contents.dev_dependencies);
    package_deps.append(&mut package_json_contents.peer_dependencies);
    package_deps.append(&mut package_json_contents.optional_dependencies);

    let deps = package_deps
        .into_iter()
        .map(|(name, version)| Package {
            name,
            language: String::from("javascript"),
            version: Some(version),
        })
        .collect();

    Ok(deps)
}

pub fn process_package_lock_json(path: &Path) -> Result<Vec<Package>> {
    let package_lock_file = path.join("package-lock.json");
    let lockfile = package_lock_json_parser::parse(read_to_string(package_lock_file)?)?;
    let package_json_deps = process_package_json(path)?;

    if let Some(lockfile_packages) = lockfile.packages {
        let mut deps = HashSet::new();

        for package_dep in package_json_deps {
            let version = lockfile_packages
                .get(&package_dep.name)
                .map_or(package_dep.version, |dep| Some(dep.version.to_string()));

            deps.insert(Package {
                version,
                ..package_dep
            });
        }

        Ok(deps.into_iter().collect())
    } else {
        Ok(package_json_deps)
    }
}

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};

    use super::*;

    #[test]
    fn it_parses_top_level_deps_from_package_lock() -> Result<()> {
        let mut expected_deps = Vec::from([
            Package {
                language: String::from("javascript"),
                name: String::from("fsevents"),
                version: Some(String::from("2.2.2")),
            },
            Package {
                language: String::from("javascript"),
                name: String::from("react"),
                version: Some(String::from("18.2.0")),
            },
            Package {
                language: String::from("javascript"),
                name: String::from("vite"),
                version: Some(String::from("5.1.4")),
            },
            Package {
                language: String::from("javascript"),
                name: String::from("zustand"),
                version: Some(String::from("4.5.1")),
            },
        ]);
        let project_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
        let test_fixtures_path = project_path.join("testdata");
        let mut deps = process_package_lock_json(test_fixtures_path.as_path())?;

        deps.sort();
        expected_deps.sort();

        assert_eq!(deps, expected_deps);
        Ok(())
    }

    #[test]
    fn it_does_not_panic() {
        let temp_dir = env::temp_dir();
        let res = process_package_lock_json(temp_dir.as_path());

        assert!(res.is_err());
    }
}
