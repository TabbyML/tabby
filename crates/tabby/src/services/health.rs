use std::{env, env::consts::ARCH, process::Command, str::from_utf8};

use anyhow::Result;
use lazy_static::lazy_static;
use nvml_wrapper::Nvml;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuExt, System, SystemExt};
use tabby_common::api::accelerator::{Accelerator, DeviceType};
use utoipa::ToSchema;

use crate::Device;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct HealthState {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_model: Option<String>,
    device: String,
    arch: String,
    cpu_info: String,
    cpu_count: usize,
    accelerators: Vec<Accelerator>,
    #[deprecated(note = "Please use the more generic accelerators instead")]
    cuda_devices: Vec<String>,
    version: Version,
}

impl HealthState {
    pub fn new(model: Option<&str>, chat_model: Option<&str>, device: &Device) -> Self {
        let (cpu_info, cpu_count) = read_cpu_info();

        let accelerators = read_accelerators();

        let mut cuda_devices = vec![];

        for accelerator in &accelerators {
            if accelerator.device_type == DeviceType::Cuda {
                cuda_devices.push(accelerator.display_name.clone());
            }
        }

        #[allow(deprecated)]
        Self {
            model: model.map(|x| x.to_owned()),
            chat_model: chat_model.map(|x| x.to_owned()),
            device: device.to_string(),
            arch: ARCH.to_string(),
            cpu_info,
            cpu_count,
            accelerators,
            cuda_devices,
            version: Version::new(),
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

pub fn read_cuda_devices() -> Result<Vec<Accelerator>> {
    // In cases of MacOS or docker containers where --gpus are not specified,
    // the Nvml::init() would return an error. In these scenarios, we
    // assign cuda_devices to be empty, indicating that the current runtime
    // environment does not support cuda interface.
    let nvml = Nvml::init()?;
    let mut cuda_devices = vec![];
    let device_count = nvml.device_count()?;
    for i in 0..device_count {
        let dev = nvml.device_by_index(i)?;
        let resource = Accelerator {
            uuid: Some(dev.uuid()?),
            chip_name: None, // NVML doesn't provide us with chip info, like "AD102" or so
            display_name: dev.name()?,
            device_type: DeviceType::Cuda,
        };
        cuda_devices.push(resource);
    }
    Ok(cuda_devices)
}

pub fn read_rocm_devices() -> Result<Vec<Accelerator>> {
    lazy_static! {
        static ref NAME_REGEX: Regex = Regex::new(r"(?m)^  Name: +([a-zA-Z0-9]+) *$",).unwrap();
        static ref MARKETING_NAME_REGEX: Regex =
            Regex::new(r"(?m)^  Marketing Name: +(\S.*\S) *$").unwrap();
        static ref UUID_REGEX: Regex =
            Regex::new(r"(?m)^  Uuid: +GPU-([a-zA-Z0-9\-]+) *$").unwrap();
        static ref DEVICE_TYPE_REGEX: Regex =
            Regex::new(r"(?m)^  Device Type: +([a-zA-Z0-9-]+) *$").unwrap();
    }

    let cmd_res = Command::new("rocminfo").output()?;
    let output = from_utf8(cmd_res.stdout.as_slice())?;
    let agent_outputs = output.split("Agent ").skip(1);
    let mut rocm_devices = vec![];
    for agent_output in agent_outputs {
        let device_type = DEVICE_TYPE_REGEX
            .captures(agent_output)
            .unwrap()
            .get(1)
            .unwrap()
            .as_str();
        if device_type != "GPU" {
            continue;
        }

        let name = NAME_REGEX
            .captures(agent_output)
            .unwrap()
            .get(1)
            .unwrap()
            .as_str();
        let marketing_name = MARKETING_NAME_REGEX
            .captures(agent_output)
            .unwrap()
            .get(1)
            .unwrap()
            .as_str();
        let uuid = UUID_REGEX
            .captures(agent_output)
            .map(|c| c.get(1).unwrap().as_str());
        let accelerator = Accelerator {
            uuid: uuid.map(|s| s.to_string()),
            chip_name: Some(name.to_string()),
            display_name: marketing_name.to_string(),
            device_type: DeviceType::Rocm,
        };
        rocm_devices.push(accelerator);
    }
    Ok(rocm_devices)
}

pub fn read_accelerators() -> Vec<Accelerator> {
    let mut devices = vec![];
    if let Ok(cuda_devices) = read_cuda_devices() {
        for dev in cuda_devices {
            devices.push(dev);
        }
    }
    if let Ok(rocm_devices) = read_rocm_devices() {
        for dev in rocm_devices {
            devices.push(dev);
        }
    }
    devices
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
