use std::sync::Arc;

use axum::Router;
use juniper::ID;
use serde::Deserialize;

use crate::schema::{
    github_repository_provider::GithubRepositoryProviderService, setting::SettingService,
};

mod github;

#[derive(Clone)]
struct IntegrationsState {
    pub settings: Arc<dyn SettingService>,
    pub github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
}

pub fn routes(
    settings: Arc<dyn SettingService>,
    github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
) -> Router {
    let state = IntegrationsState {
        settings,
        github_repository_provider,
    };
    Router::new().nest("/github", github::routes(state))
}

#[derive(Deserialize)]
struct CallbackParams {
    state: ID,
    code: String,
}
