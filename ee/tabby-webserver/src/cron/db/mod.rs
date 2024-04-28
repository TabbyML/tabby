//! db maintenance jobs

mod github;
mod gitlab;

use std::sync::Arc;

use super::controller::JobController;
use crate::schema::{
    auth::AuthenticationService,
    repository::{GithubRepositoryService, GitlabRepositoryService},
};

const EVERY_TWO_HOURS: &str = "0 0 1/2 * * * *";
const EVERY_TEN_MINUTES: &str = "0 1/10 * * * *";

pub async fn register_jobs(
    controller: &mut JobController,
    auth: Arc<dyn AuthenticationService>,
    github: Arc<dyn GithubRepositoryService>,
    gitlab: Arc<dyn GitlabRepositoryService>,
) {
    let cloned_auth = auth.clone();
    controller
        .register("remove_staled_refresh_token", EVERY_TWO_HOURS, move || {
            let auth = cloned_auth.clone();
            Box::pin(async move { Ok(auth.delete_expired_token().await?) })
        })
        .await;

    let cloned_auth = auth.clone();
    controller
        .register("remove_staled_password_reset", EVERY_TWO_HOURS, move || {
            let auth = cloned_auth.clone();
            Box::pin(async move { Ok(auth.delete_expired_password_resets().await?) })
        })
        .await;

    controller
        .register_public("github_repositories", EVERY_TEN_MINUTES, move |context| {
            let context = context.clone();
            let github = github.clone();
            Box::pin(async move { github::refresh_all_repositories(context, github).await })
        })
        .await;

    controller
        .register_public("gitlab_repositories", EVERY_TEN_MINUTES, move |context| {
            let gitlab = gitlab.clone();
            let context = context.clone();
            Box::pin(async move { gitlab::refresh_all_repositories(context, gitlab).await })
        })
        .await;
}
