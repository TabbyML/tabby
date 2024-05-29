#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;

    use serde_json::json;
    use tabby_common::{config::Config, path::set_tabby_root};
    use temp_testdir::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[tokio::test]
    async fn end_to_end() {
        let root = TempDir::default();
        create_dir_all(&root).expect("Failed to create tabby root");
        set_tabby_root(root.to_path_buf());
        let config: Config = serde_json::from_value(json!({
                "repositories": [{
                    "git_url": "https://github.com/TabbyML/interview-questions"
        }]
        }))
        .unwrap();

        tabby_scheduler::scheduler(true, &config).await;
    }
}
