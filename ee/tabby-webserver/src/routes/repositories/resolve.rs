use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use axum::{body::Body, response::Response};
use hyper::StatusCode;
use juniper::ID;
use serde::Deserialize;
use tabby_common::config::RepositoryConfig;
use tabby_schema::repository::{RepositoryKind, RepositoryService};

#[derive(Deserialize, Debug)]
pub struct ResolveParams {
    pub kind: RepositoryKind,
    pub id: ID,
    rev: Option<String>,
    path: Option<String>,
}

pub(super) struct ResolveState {
    service: Arc<dyn RepositoryService>,
    config: Vec<RepositoryConfig>,
}

impl ResolveState {
    pub fn new(service: Arc<dyn RepositoryService>, config: Vec<RepositoryConfig>) -> Self {
        Self { service, config }
    }

    async fn find_repository(&self, params: &ResolveParams) -> Option<PathBuf> {
        if let Some(index) = params.id.strip_prefix("CONFIG_") {
            let index: usize = index.parse().ok()?;
            return Some(self.config.get(index)?.dir());
        }

        let repository = self
            .service
            .resolve_repository(&params.kind, &params.id)
            .await
            .ok()?;
        Some(repository.dir)
    }

    pub async fn resolve(&self, params: ResolveParams) -> Result<Response<Body>, StatusCode> {
        let Some(root) = self.find_repository(&params).await else {
            return Err(StatusCode::NOT_FOUND);
        };

        tabby_git::serve_file(&root, params.rev.as_deref(), params.path.as_deref())
    }
}
