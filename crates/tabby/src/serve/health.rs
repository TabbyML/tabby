use std::{env::consts::ARCH, sync::Arc};

use axum::{extract::State, Json};
use rust_gpu_tools::Device;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuExt, System, SystemExt};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
struct CPUInfo {
    brand: String,
    number: usize,
}

impl CPUInfo {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_cpu();
        let cpus = sys.cpus();
        let brand = if cpus.len() > 0 {
            let cpu = &cpus[0];
            cpu.brand().to_string()
        } else {
            "no cpus assigned".to_string()
        };
        Self {
            brand,
            number: cpus.len(),
        }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
struct GPUInfo {
    gpu_list: Vec<String>,
}

impl GPUInfo {
    pub fn new() -> Self {
        let mut gpu_list = vec![];
        let devices = Device::all();
        for device in devices.iter() {
            gpu_list.push(device.name());
        }
        Self { gpu_list }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct HealthState {
    model: String,
    device: String,
    compute_type: String,
    architecture_info: String,
    cpu_info: CPUInfo,
    gpu_info: GPUInfo,
}

impl HealthState {
    pub fn new(args: &super::ServeArgs) -> Self {
        Self {
            model: args.model.clone(),
            device: args.device.to_string(),
            compute_type: args.compute_type.to_string(),
            architecture_info: ARCH.to_string(),
            cpu_info: CPUInfo::new(),
            gpu_info: GPUInfo::new(),
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
