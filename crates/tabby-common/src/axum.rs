use axum::http::HeaderName;
use axum_extra::headers::Header;

use crate::{config::CodeRepository, constants::USER_HEADER_FIELD_NAME};

#[derive(Debug)]
pub struct MaybeUser(pub Option<String>);

pub static USER_HEADER: HeaderName = HeaderName::from_static(USER_HEADER_FIELD_NAME);

impl Header for MaybeUser {
    fn name() -> &'static axum::http::HeaderName {
        &USER_HEADER
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, axum_extra::headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i axum::http::HeaderValue>,
    {
        let Some(value) = values.next() else {
            return Ok(MaybeUser(None));
        };
        let str = value.to_str().expect("User email is always a valid string");
        Ok(MaybeUser(Some(str.to_string())))
    }

    fn encode<E: Extend<axum::http::HeaderValue>>(&self, _values: &mut E) {
        todo!()
    }
}

#[derive(Debug, Default, Clone)]
pub struct AllowedCodeRepository {
    list: Vec<CodeRepository>,
}

impl AllowedCodeRepository {
    pub fn new(list: Vec<CodeRepository>) -> Self {
        Self { list }
    }

    pub fn new_from_config() -> Self {
        let list = crate::config::Config::load()
            .map(|x| {
                x.repositories
                    .into_iter()
                    .enumerate()
                    .map(|(i, repo)| {
                        CodeRepository::new(repo.git_url(), &crate::config::config_index_to_id(i))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self { list }
    }
    pub fn closest_match(&self, git_url: &str) -> Option<&str> {
        closest_match(git_url, self.list.iter())
    }
}

fn closest_match<'a>(
    git_url: &str,
    repositories: impl IntoIterator<Item = &'a CodeRepository>,
) -> Option<&'a str> {
    let git_search = parse_git_url::GitUrl::parse(git_url).ok()?;

    repositories
        .into_iter()
        .filter(|elem| {
            parse_git_url::GitUrl::parse(&elem.git_url).is_ok_and(|x| x.name == git_search.name)
        })
        // If there're multiple matches, we pick the one with highest alphabetical order
        .min_by_key(|elem| elem.canonical_git_url())
        .map(|x| x.source_id.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_match_first {
        ($query:literal, $candidates:expr) => {
            let candidates: Vec<_> = $candidates
                .into_iter()
                .enumerate()
                .map(|(i, x)| CodeRepository::new(&x, &crate::config::config_index_to_id(i)))
                .collect();
            let expect = &candidates[0];
            assert_eq!(
                closest_match($query, &candidates),
                Some(expect.source_id.as_ref())
            );
        };
    }

    macro_rules! assert_match_none {
        ($query:literal, $candidates:expr) => {
            let candidates: Vec<_> = $candidates
                .into_iter()
                .enumerate()
                .map(|(i, x)| CodeRepository::new(&x, &crate::config::config_index_to_id(i)))
                .collect();
            assert_eq!(closest_match($query, &candidates), None);
        };
    }

    #[test]
    fn test_closest_match() {
        // Test .git suffix should still match
        assert_match_first!(
            "https://github.com/example/test.git",
            ["https://github.com/example/test"]
        );

        // Test auth in URL should still match
        assert_match_first!(
            "https://creds@github.com/example/test",
            ["https://github.com/example/test"]
        );

        // Test name must be exact match
        assert_match_none!(
            "https://github.com/example/another-repo",
            ["https://github.com/example/anoth-repo"]
        );

        // Test different repositories with a common prefix should not match
        assert_match_none!(
            "https://github.com/TabbyML/tabby",
            ["https://github.com/TabbyML/registry-tabby"]
        );

        // Test entirely different repository names should not match
        assert_match_none!(
            "https://github.com/TabbyML/tabby",
            ["https://github.com/TabbyML/uptime"]
        );

        assert_match_none!("https://github.com", ["https://github.com/TabbyML/tabby"]);

        // Test different host
        assert_match_first!(
            "https://bitbucket.com/TabbyML/tabby",
            ["https://github.com/TabbyML/tabby"]
        );

        // Test multiple close matches
        assert_match_none!(
            "git@github.com:TabbyML/tabby",
            [
                "https://bitbucket.com/CrabbyML/crabby",
                "https://gitlab.com/TabbyML/registry-tabby",
            ]
        );
    }

    #[test]
    fn test_closest_match_url_format_differences() {
        // Test different protocol and suffix should still match
        assert_match_first!(
            "git@github.com:TabbyML/tabby.git",
            ["https://github.com/TabbyML/tabby"]
        );

        // Test different protocol should still match
        assert_match_first!(
            "git@github.com:TabbyML/tabby",
            ["https://github.com/TabbyML/tabby"]
        );

        // Test URL without organization should still match
        assert_match_first!(
            "https://custom-git.com/tabby",
            ["https://custom-git.com/TabbyML/tabby"]
        );
    }

    #[test]
    fn test_closest_match_local_url() {
        assert_match_first!(
            "git@github.com:TabbyML/tabby.git",
            ["file:///home/TabbyML/tabby"]
        );
    }
}
