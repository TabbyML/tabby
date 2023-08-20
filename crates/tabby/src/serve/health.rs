use std::{env::consts::ARCH, sync::Arc};

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use sysinfo::{CpuExt, System, SystemExt};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct HealthState {
    model: String,
    device: String,
    compute_type: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    version: Version,
}

impl HealthState {
    pub fn new(args: &super::ServeArgs) -> Self {
        let mut sys = System::new_all();
        sys.refresh_cpu();
        let cpus = sys.cpus();
        let cpu_info = if !cpus.is_empty() {
            let cpu = &cpus[0];
            cpu.brand().to_string()
        } else {
            "unknown".to_string()
        };

        Self {
            model: args.model.clone(),
            device: args.device.to_string(),
            compute_type: args.compute_type.to_string(),
            arch: ARCH.to_string(),
            cpu_info,
            cpu_count: cpus.len(),
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
