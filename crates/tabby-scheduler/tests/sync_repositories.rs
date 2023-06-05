use tabby_scheduler::*;
use temp_testdir::*;

use tabby_common::config::{Config, Repository};
use tabby_common::path::set_tabby_root;

#[test]
fn it_works() {
    set_tabby_root(TempDir::default().to_path_buf());

    let config = Config {
        repositories: vec![Repository {
            git_url: "https://github.com/TabbyML/interview-questions".to_owned(),
        }],
    };

    config.save();
    job_sync_repositories()
}
