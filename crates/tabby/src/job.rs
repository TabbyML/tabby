use tabby_common::config::{Config, RepositoryConfig};
use tabby_webserver::public::{ConnectHubRequest, RepositoryAccess};

pub async fn get_repositories(
    url: Option<String>,
    token: Option<String>,
    config: &[RepositoryConfig],
) -> Vec<RepositoryConfig> {
    match url.zip(token) {
        Some((addr, token)) => {
            let client =
                tabby_webserver::public::create_client(&addr, &token, ConnectHubRequest::Job).await;

            RepositoryAccess::get_repositories(&client)
                .await
                .expect("Must be able to load repositories")
        }
        None => config.to_vec(),
    }
}

pub async fn start_sync_job(args: JobArgs, config: &Config) {
    let repositories = get_repositories(args.url, args.token, &config.repositories).await;
    tabby_scheduler::job_sync(&repositories)
}

pub async fn start_index_job(args: JobArgs, config: &Config) {
    let repositories = get_repositories(args.url, args.token, &config.repositories).await;
    tabby_scheduler::job_index(&repositories)
}

#[derive(clap::Args)]
pub struct JobArgs {
    #[clap(long, requires = "url")]
    pub token: Option<String>,
    #[clap(long, requires = "token")]
    pub url: Option<String>,
}
