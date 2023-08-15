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
    cpu_count: u32,
}

impl HealthState {
    pub fn new(args: &super::ServeArgs) -> Self {
        let mut sys = System::new_all();
        sys.refresh_cpu();
        let cpus = sys.cpus();
        let cpu_info = if cpus.len() > 0 {
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
            cpu_count: cpus.len() as u32,
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
