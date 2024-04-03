use std::path::Path;

use anyhow::Result;
use cargo_lock::dependency::graph::EdgeDirection;
use tabby_common::Package;

fn extract_deps<'a, I>(packages: I) -> Vec<Package>
where
    I: IntoIterator<Item = &'a cargo_lock::Package>,
{
    let mut res = packages
        .into_iter()
        .map(|package| Package {
            language: String::from("rust"),
            name: package.name.to_string(),
            version: Some(package.version.to_string()),
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    res.sort_unstable();
    res
}

pub fn process_cargo(path: &Path) -> Result<Vec<Package>> {
    let cargo_lock_file = path.join("Cargo.lock");

    let lockfile = cargo_lock::Lockfile::load(cargo_lock_file)?;

    let tree = lockfile.dependency_tree()?;
    let graph = tree.graph();

    let root_pkg_idx = graph
        .externals(EdgeDirection::Incoming)
        .collect::<std::collections::HashSet<_>>();
    let direct_deps_idx = root_pkg_idx
        .iter()
        .flat_map(|idx| graph.neighbors_directed(*idx, EdgeDirection::Outgoing))
        .collect::<std::collections::HashSet<_>>();
    let deps = extract_deps(direct_deps_idx.iter().map(|dep_idx| &graph[*dep_idx]));
    Ok(deps)
}
