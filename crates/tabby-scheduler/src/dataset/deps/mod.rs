mod python;
mod rust;

use std::{collections::HashSet, path::Path};

use tabby_common::DependencyFile;

pub fn collect(path: &Path, file: &mut DependencyFile) {
    if let Ok(mut deps) = python::process_requirements_txt(path) {
        file.direct.append(&mut deps);
    }

    if let Ok(mut deps) = rust::process_cargo(path) {
        file.direct.append(&mut deps);
    }

    // Remove duplicates across sources.
    let deps = file.direct.clone().into_iter().collect::<HashSet<_>>();
    file.direct = deps.into_iter().collect();
}
