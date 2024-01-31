use tabby_common::config::RepositoryConfig;
use tabby_webserver::public::{ConnectHubRequest, Context};

pub async fn get_repositories(
    url: Option<String>,
    token: Option<String>,
    config: &Vec<RepositoryConfig>,
) -> Vec<RepositoryConfig> {
    match url.zip(token) {
        Some((addr, token)) => {
            let client =
                tabby_webserver::public::create_client(&addr, &token, ConnectHubRequest::Job).await;
            
            client
                .get_repositories(Context::current())
                .await
                .expect("Must be able to load repositories")
        }
        None => config.to_vec(),
    }
}
