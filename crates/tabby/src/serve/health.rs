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
    device: String,
    compute_type: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    gpu_info: Vec<String>,
    version: Version,
}

impl HealthState {
    pub fn new(args: &super::ServeArgs) -> Self {
        let cpu_stats_tuple = get_cpu_stats();

        let gpu_info_res = get_gpu_stats();
        let gpu_info = match gpu_info_res {
            Ok(s) => s,
            Err(_) => vec![],
        };

        Self {
            model: args.model.clone(),
            device: args.device.to_string(),
            compute_type: args.compute_type.to_string(),
            arch: ARCH.to_string(),
            cpu_info: cpu_stats_tuple.0,
            cpu_count: cpu_stats_tuple.1,
            gpu_info,
            version: Version::new(),
        }
    }
}

pub struct CPUStat {
    pub info: String,
    pub count: usize,
}

fn get_cpu_stats() -> (String, usize) {
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

fn get_gpu_stats() -> Result<Vec<String>> {
    // In cases of MacOS or docker containers where --gpus are not specified,
    // the Nvml::init() would return an error. In these scenarios, we
    // assign gpu_info to be empty, indicating that the current runtime
    // environment does not support cuda interface.
    let nvml = Nvml::init()?;
    let mut gpu_info = vec![];
    let device_count = nvml.device_count()?;
    for i in 0..device_count {
        let name = nvml.device_by_index(i)?.name()?;
        gpu_info.push(name);
    }
    Ok(gpu_info)
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
