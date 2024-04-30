//! db maintenance jobs

mod github;
mod gitlab;

use std::sync::Arc;

use super::{controller::JobController, every_two_hours};
use crate::schema::{
    auth::AuthenticationService,
    repository::{GithubRepositoryService, GitlabRepositoryService},
};

pub async fn register_jobs(
    controller: &mut JobController,
    auth: Arc<dyn AuthenticationService>,
    github: Arc<dyn GithubRepositoryService>,
    gitlab: Arc<dyn GitlabRepositoryService>,
) {
    let cloned_auth = auth.clone();
    controller
        .register(
            "remove_staled_refresh_token",
            &every_two_hours(),
            move || {
                let auth = cloned_auth.clone();
                Box::pin(async move { Ok(auth.delete_expired_token().await?) })
            },
        )
        .await;

    let cloned_auth = auth.clone();
    controller
        .register(
            "remove_staled_password_reset",
            &every_two_hours(),
            move || {
                let auth = cloned_auth.clone();
                Box::pin(async move { Ok(auth.delete_expired_password_resets().await?) })
            },
        )
        .await;

    controller
        .register_public("github_repositories", &every_two_hours(), move |context| {
            let context = context.clone();
            let github = github.clone();
            Box::pin(async move { github::refresh_all_repositories(context, github).await })
        })
        .await;

    controller
        .register_public("gitlab_repositories", &every_two_hours(), move |context| {
            let gitlab = gitlab.clone();
            let context = context.clone();
            Box::pin(async move { gitlab::refresh_all_repositories(context, gitlab).await })
        })
        .await;
}
