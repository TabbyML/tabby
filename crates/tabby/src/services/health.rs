use std::env::consts::ARCH;

use anyhow::Result;
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuExt, System, SystemExt};
use utoipa::ToSchema;

use crate::Device;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct HealthState {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_device: Option<String>,
    device: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    cuda_devices: Vec<String>,
    version: Version,
    webserver: Option<bool>,
}

impl HealthState {
    pub fn new(
        model: Option<&str>,
        device: &Device,
        chat_model: Option<&str>,
        chat_device: Option<&Device>,
        webserver: Option<bool>,
    ) -> Self {
        let (cpu_info, cpu_count) = read_cpu_info();

        let cuda_devices = match read_cuda_devices() {
            Ok(s) => s,
            Err(_) => vec![],
        };

        let http_model_name = Some("Remote");
        let is_model_http = device == &Device::ExperimentalHttp;
        let model = if is_model_http {
            http_model_name
        } else {
            model
        };

        let is_chat_model_http = chat_device == Some(&Device::ExperimentalHttp);
        let chat_model = if is_chat_model_http {
            http_model_name
        } else {
            chat_model
        };

        Self {
            model: model.map(|x| x.to_string()),
            chat_model: chat_model.map(|x| x.to_owned()),
            chat_device: chat_device.map(|x| x.to_string()),
            device: device.to_string(),
            arch: ARCH.to_string(),
            cpu_info,
            cpu_count,
            cuda_devices,
            version: Version::new(),
            webserver,
        }
    }
}

pub fn read_cpu_info() -> (String, usize) {
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

pub fn read_cuda_devices() -> Result<Vec<String>> {
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
