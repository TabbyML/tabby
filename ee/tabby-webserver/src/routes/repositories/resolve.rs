use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use axum::{body::Body, response::Response};
use hyper::StatusCode;
use juniper::ID;
use serde::{Deserialize};
use tabby_schema::repository::{RepositoryKind, RepositoryService};
use tabby_search::ServeGit;
use tracing::warn;

#[derive(Deserialize, Debug)]
pub struct ResolveParams {
    pub kind: RepositoryKind,
    pub id: ID,
    path: Option<String>,
}

pub(super) struct ResolveState {
    service: Arc<dyn RepositoryService>,
}

impl ResolveState {
    pub fn new(service: Arc<dyn RepositoryService>) -> Self {
        Self { service }
    }

    async fn find_repository(&self, params: &ResolveParams) -> Option<PathBuf> {
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

        let serve = ServeGit::new(&root).map_err(|e| {
            warn!("Failed to open repository: {:?}", e);
            StatusCode::NOT_FOUND
        })?;

        serve.serve(None, params.path.as_deref())
    }
}
