use std::{env::consts::ARCH, sync::Arc};

use anyhow::Result;
use axum::{extract::State, Json};
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuExt, System, SystemExt};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct HealthState {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instruct_model: Option<String>,
    device: String,
    compute_type: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    cuda_devices: Vec<String>,
    version: Version,
}

impl HealthState {
    pub fn new(args: &super::ServeArgs) -> Self {
        let (cpu_info, cpu_count) = read_cpu_info();

        let cuda_devices = match read_cuda_devices() {
            Ok(s) => s,
            Err(_) => vec![],
        };

        Self {
            model: args.model.clone(),
            instruct_model: args.instruct_model.clone(),
            device: args.device.to_string(),
            compute_type: args.compute_type.to_string(),
            arch: ARCH.to_string(),
            cpu_info,
            cpu_count,
            cuda_devices,
            version: Version::new(),
        }
    }
}

fn read_cpu_info() -> (String, usize) {
    let mut system = System::new_all();
    system.refresh_cpu();
    let cpus = system.cpus();
    let count = cpus.len();
    let info = if count > 0 {
        let cpu = &cpus[0];
        cpu.brand().to_string()
    } else {
        "unknown".to_string()
    };

    (info, count)
}

fn read_cuda_devices() -> Result<Vec<String>> {
    // In cases of MacOS or docker containers where --gpus are not specified,
    // the Nvml::init() would return an error. In these scenarios, we
    // assign cuda_devices to be empty, indicating that the current runtime
    // environment does not support cuda interface.
    let nvml = Nvml::init()?;
    let mut cuda_devices = vec![];
    let device_count = nvml.device_count()?;
    for i in 0..device_count {
        let name = nvml.device_by_index(i)?.name()?;
        cuda_devices.push(name);
    }
    Ok(cuda_devices)
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
