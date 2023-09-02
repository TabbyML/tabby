use tabby_scheduler;

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
    #[tokio::test]
    async fn end_to_end() {
        set_tabby_root(TempDir::default().to_path_buf());

        let config = Config {
            repositories: vec![Repository {
                git_url: "https://github.com/TabbyML/interview-questions".to_owned(),
            }],
            experimental: Experimental::default(),
        };

        let res = tabby_scheduler::scheduler(true, Some(config)).await;
        assert!(res.is_ok());
    }
}
