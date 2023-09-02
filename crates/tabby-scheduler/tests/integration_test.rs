use tabby_scheduler::dataset;
use tabby_scheduler::index;
use tabby_scheduler::repository;

#[cfg(test)]
mod tests {
    use tabby_common::{
        config::{Config, Experimental, Repository},
        path::set_tabby_root,
    };
    use temp_testdir::*;
    use tracing_test::traced_test;

    use super::*;

    #[traced_test]
    #[test]
    fn end_to_end() {
        set_tabby_root(TempDir::default().to_path_buf());

        let config = Config {
            repositories: vec![Repository {
                git_url: "https://github.com/TabbyML/interview-questions".to_owned(),
            }],
            experimental: Experimental::default(),
        };

        repository::sync_repositories(&config).unwrap();
        dataset::create_dataset(&config).unwrap();
        index::index_repositories(&config).unwrap();
    }
}