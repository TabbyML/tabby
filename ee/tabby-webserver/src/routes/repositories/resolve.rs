use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use axum::{body::Body, response::Response};
use hyper::StatusCode;
use juniper::ID;
use serde::Deserialize;
use tabby_schema::{
    policy::AccessPolicy,
    repository::{RepositoryKind, RepositoryService},
};

#[derive(Deserialize, Debug)]
pub struct ResolveParams {
    pub kind: RepositoryKind,
    pub id: ID,
    rev: Option<String>,
    path: Option<String>,
}

pub(super) struct ResolveState {
    service: Arc<dyn RepositoryService>,
}

impl ResolveState {
    pub fn new(service: Arc<dyn RepositoryService>) -> Self {
        Self { service }
    }

    async fn find_repository(
        &self,
        policy: &AccessPolicy,
        params: &ResolveParams,
    ) -> Option<PathBuf> {
        let repository = self
            .service
            .resolve_repository(policy, &params.kind, &params.id)
            .await
            .ok()?;
        Some(repository.dir)
    }

    pub async fn resolve(
        &self,
        policy: &AccessPolicy,
        params: ResolveParams,
    ) -> Result<Response<Body>, StatusCode> {
        let Some(root) = self.find_repository(policy, &params).await else {
            return Err(StatusCode::NOT_FOUND);
        };

        tabby_git::serve_file(&root, params.rev.as_deref(), params.path.as_deref())
    }
}
