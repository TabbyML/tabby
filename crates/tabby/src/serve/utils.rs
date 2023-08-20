use anyhow::{anyhow, Result};
use nvml_wrapper::Nvml;

// TabbyContext is the struct for global shared context
// to be used by router app.
pub struct TabbyContext {
    pub gpu_stats_manager: GPUStatsManager,
}

impl TabbyContext {
    pub fn new() -> Self {
        let gpu_stats_manager = GPUStatsManager::init();
        Self {
            gpu_stats_manager,
        }
    }
}

struct GPUStatsManager {
    // In compilation envs where there is no cuda,
    // We do not use the nvml, therefore setting it as None.
    nvml: Option<Nvml>,
}

#[cfg(not(target_os = "macos"))]
impl GPUStatsManager {
    pub fn init() -> Self {
        let nvml = Nvml::init().unwrap();
        Self {
            nvml: Some(nvml),
        }
    }

    pub fn get_gpu_stats(self: &Self) -> Result<Vec<String>> {
        let mut res = vec![];
        let nvml = &self.nvml.unwrap();
        let device_count = nvml.device_count()?;
        for i in 0..device_count {
            let name = nvml.device_by_index(i)?.name()?;
            res.push(name);
        }
        Ok(res)
    }
}

#[cfg(target_os = "macos")]
impl GPUStatsManager {
    pub fn init() -> Self {
        Self {
            nvml: None,
        }
    }

    pub fn get_gpu_stats() -> Result<Vec<String>> {
        Ok(vec![])
    }
}