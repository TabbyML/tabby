use std::{env::consts::ARCH, sync::Arc};

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::utils::TabbyContext;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct HealthState {
    model: String,
    device: String,
    compute_type: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    gpu_info: Vec<String>,
    version: Version,
}

impl HealthState {
    pub fn new(args: &super::ServeArgs, context: &mut TabbyContext) -> Self {
        let cpu_stats = context.cpu_stats_manager.get_stats();

        let gpu_info_res = context.gpu_stats_manager.get_stats();
        let gpu_info = match gpu_info_res {
            Ok(s) => s,
            Err(_) => vec![],
        };

        Self {
            model: args.model.clone(),
            device: args.device.to_string(),
            compute_type: args.compute_type.to_string(),
            arch: ARCH.to_string(),
            cpu_info: cpu_stats.info,
            cpu_count: cpu_stats.count,
            gpu_info,
            version: Version::new(),
        }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Version {
    build_date: String,
    build_timestamp: String,
    git_sha: String,
    git_describe: String,
}

impl Version {
    fn new() -> Self {
        Self {
            build_date: env!("VERGEN_BUILD_DATE").to_string(),
            build_timestamp: env!("VERGEN_BUILD_TIMESTAMP").to_string(),
            git_sha: env!("VERGEN_GIT_SHA").to_string(),
            git_describe: env!("VERGEN_GIT_DESCRIBE").to_string(),
        }
    }
}

#[utoipa::path(
    post,
    path = "/v1/health",
    tag = "v1",
    responses(
        (status = 200, description = "Success", body = HealthState, content_type = "application/json"),
    )
)]
pub async fn health(State(state): State<Arc<HealthState>>) -> Json<HealthState> {
    Json(state.as_ref().clone())
}
