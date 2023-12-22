#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;

    use tabby_common::{
        config::{Config, RepositoryConfig, ServerConfig},
        path::set_tabby_root,
    };
    use temp_testdir::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[tokio::test]
    async fn end_to_end() {
        let root = TempDir::default();
        create_dir_all(&root).expect("Failed to create tabby root");
        set_tabby_root(root.to_path_buf());

        let config = Config {
            repositories: vec![RepositoryConfig {
                name: None,
                git_url: "https://github.com/TabbyML/interview-questions".to_owned(),
            }],
            server: ServerConfig::default(),
        };

        let res = tabby_scheduler::scheduler(true, &config).await;
        res.expect("Failed to run scheduler");
    }
}
