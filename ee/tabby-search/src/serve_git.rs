use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use git2::Blob;
use serde::Serialize;

const DIRECTORY_MIME_TYPE: &str = "application/vnd.directory+json";

fn resolve<'a>(
    repo: &'a git2::Repository,
    rev: Option<&str>,
    relpath_str: Option<&str>,
) -> anyhow::Result<Resolve<'a>> {
    let commit = if let Some(rev) = rev {
        let reference = repo.revparse_single(rev)?;
        reference.peel_to_commit()?
    } else {
        repo.head()?.peel_to_commit()?
    };
    let tree = commit.tree()?;

    let relpath = Path::new(relpath_str.unwrap_or(""));
    let object = if relpath_str.is_some() {
        tree.get_path(relpath)?.to_object(repo)?
    } else {
        tree.as_object().clone()
    };

    match object.kind() {
        Some(git2::ObjectType::Blob) => {
            let blob = object.as_blob().context("failed to resolve blob")?;
            Ok(Resolve::File(relpath.to_owned(), blob.clone()))
        }
        Some(git2::ObjectType::Tree) => Ok(Resolve::Dir(
            object
                .as_tree()
                .context("failed to resolve tree")?
                .iter()
                .map(|entry| {
                    let kind = if entry.kind() == Some(git2::ObjectType::Tree) {
                        DirEntryKind::Dir
                    } else {
                        DirEntryKind::File
                    };
                    DirEntry {
                        kind,
                        basename: relpath
                            .join(entry.name().expect("failed to resolve entry name"))
                            .display()
                            .to_string(),
                    }
                })
                .collect::<Vec<_>>(),
        )),
        _ => {
            bail!("unsupported object type");
        }
    }
}

pub fn serve(
    repository: &git2::Repository,
    rev: Option<&str>,
    relpath: Option<&str>,
) -> std::result::Result<Response<Body>, StatusCode> {
    let resolve = match resolve(repository, rev, relpath) {
        Ok(resolve) => resolve,
        Err(_) => {
            return Err(StatusCode::NOT_FOUND);
        }
    };

    let resp = match resolve {
        Resolve::Dir(entries) => {
            let json =
                serde_json::to_string(&ListDir { entries }).expect("failed to serialize response");
            Response::builder()
                .header(header::CONTENT_TYPE, DIRECTORY_MIME_TYPE)
                .body(Body::from(json))
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        }
        Resolve::File(path, blob) => {
            let body = Body::from(blob.content().to_owned());
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Response::builder()
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(body)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        }
    };

    Ok(resp)
}

#[derive(Debug)]
pub enum Resolve<'a> {
    Dir(Vec<DirEntry>),
    File(PathBuf, Blob<'a>),
}

#[derive(Serialize)]
struct ListDir {
    entries: Vec<DirEntry>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum DirEntryKind {
    File,
    Dir,
}

#[derive(Serialize, Debug)]
pub struct DirEntry {
    kind: DirEntryKind,
    basename: String,
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;
    use crate::testutils::TempGitRepository;

    #[test]
    fn test_resolve() {
        let root = TempGitRepository::default();
        let repo = root.repository();

        assert_matches!(resolve(&repo, None, None), Ok(Resolve::Dir(_)));
        assert_matches!(
            resolve(&repo, None, Some("README.md")),
            Ok(Resolve::File(_, _))
        );
    }

    #[test]
    fn test_serve() {
        let root = TempGitRepository::default();
        let repo = root.repository();

        assert_matches!(serve(&repo, None, None), Ok(_));
        assert_matches!(serve(&repo, None, Some("README.md")), Ok(_));
        assert_matches!(
            serve(&repo, None, Some("NotExists")),
            Err(StatusCode::NOT_FOUND)
        );
    }
}
