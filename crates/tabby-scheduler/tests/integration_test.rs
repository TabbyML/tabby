use async_trait::async_trait;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};

struct StaticRepositoryAccess {
    repositories: Vec<RepositoryConfig>,
}

#[async_trait]
impl RepositoryAccess for StaticRepositoryAccess {
    async fn list_repositories(&self) -> anyhow::Result<Vec<RepositoryConfig>> {
        Ok(self.repositories.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;

    use tabby_common::{config::RepositoryConfig, path::set_tabby_root};
    use temp_testdir::*;
    use tracing_test::traced_test;

    use super::StaticRepositoryAccess;

    #[traced_test]
    #[tokio::test]
    async fn end_to_end() {
        let root = TempDir::default();
        create_dir_all(&root).expect("Failed to create tabby root");
        set_tabby_root(root.to_path_buf());

        let config = StaticRepositoryAccess {
            repositories: vec![RepositoryConfig::new(
                "https://github.com/TabbyML/interview-questions".to_owned(),
            )],
        };

        let res = tabby_scheduler::scheduler(true, config, &[]).await;
        res.expect("Failed to run scheduler");
    }
}
