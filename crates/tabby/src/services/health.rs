use std::env::consts::ARCH;

use anyhow::Result;
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use sysinfo::System;
use tabby_common::config::{ModelConfig, ModelConfigGroup};
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
    cuda_devices: Vec<String>,

    // Model health status; the above fields are slated for future deprecation.
    models: ModelsHealth,

    // CPU information for Tabby server
    arch: String,
    cpu_info: String,
    cpu_count: usize,

    version: Version,
    webserver: Option<bool>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ModelsHealth {
    #[serde(skip_serializing_if = "Option::is_none")]
    completion: Option<ModelHealth>,

    #[serde(skip_serializing_if = "Option::is_none")]
    chat: Option<ModelHealth>,

    embedding: ModelHealth,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
enum ModelHealth {
    #[serde(rename = "remote")]
    Remote(RemoteModelHealth),
    #[serde(rename = "local")]
    Local(LocalModelHealth),
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct RemoteModelHealth {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
    api_endpoint: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct LocalModelHealth {
    model_id: String,
    device: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cuda_devices: Vec<String>,
}

impl From<&ModelConfig> for ModelHealth {
    fn from(model_config: &ModelConfig) -> Self {
        match model_config {
            ModelConfig::Http(http) => ModelHealth::Remote(RemoteModelHealth {
                kind: http.kind.clone(),
                model_name: http.model_name.clone(),
                api_endpoint: http.api_endpoint.clone().unwrap_or_default(),
            }),
            ModelConfig::Local(llama) => ModelHealth::Local(LocalModelHealth {
                model_id: llama.model_id.clone(),
                device: String::new(),
                cuda_devices: vec![],
            }),
        }
    }
}

impl From<&ModelConfigGroup> for ModelsHealth {
    fn from(model_config: &ModelConfigGroup) -> Self {
        let completion = model_config.completion.as_ref().map(ModelHealth::from);
        let chat = model_config.chat.as_ref().map(ModelHealth::from);

        let embedding = ModelHealth::from(&model_config.embedding);

        Self {
            completion,
            chat,
            embedding,
        }
    }
}

impl HealthState {
    pub fn new(
        model_config: &ModelConfigGroup,
        device: &Device,
        chat_device: Option<&Device>,
        webserver: Option<bool>,
    ) -> Self {
        let (cpu_info, cpu_count) = read_cpu_info();

        let cuda_devices = read_cuda_devices().unwrap_or_default();
        let mut models = ModelsHealth::from(model_config);
        if let Some(model) = &mut models.completion {
            if let ModelHealth::Local(ref mut local) = model {
                local.device = device.to_string();
                local.cuda_devices = cuda_devices.clone();
            }
        }
        if let Some(model) = &mut models.chat {
            if let ModelHealth::Local(ref mut local) = model {
                local.device = chat_device.unwrap_or(device).to_string();
                local.cuda_devices = cuda_devices.clone();
            }
        }
        if let ModelHealth::Local(ref mut local) = models.embedding {
            local.device = device.to_string();
            local.cuda_devices = cuda_devices.clone();
        }

        Self {
            model: to_model_name(&model_config.completion),
            chat_model: to_model_name(&model_config.chat),
            chat_device: chat_device.map(|x| x.to_string()),
            device: device.to_string(),
            models,
            arch: ARCH.to_string(),
            cpu_info,
            cpu_count,
            cuda_devices,
            version: Version::new(),
            webserver,
        }
    }
}

fn to_model_name(model: &Option<ModelConfig>) -> Option<String> {
    if let Some(model) = model {
        match model {
            ModelConfig::Http(http) => http
                .model_name
                .clone()
                .or_else(|| Some("Remote".to_string())),
            ModelConfig::Local(llama) => Some(llama.model_id.clone()),
        }
    } else {
        None
    }
}

pub fn read_cpu_info() -> (String, usize) {
    let mut system = System::new_all();
    system.refresh_cpu_all();
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
