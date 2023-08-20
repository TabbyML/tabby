use anyhow::Result;
use nvml_wrapper::Nvml;
use sysinfo::{CpuExt, System, SystemExt};

// TabbyContext is the struct for global shared context
// to be used by router app.
pub struct TabbyContext {
    pub cpu_stats_manager: CPUStatsManager,
    pub gpu_stats_manager: GPUStatsManager,
}

impl TabbyContext {
    pub fn new() -> Self {
        let cpu_stats_manager = CPUStatsManager::init();
        let gpu_stats_manager = GPUStatsManager::init();
        Self {
            cpu_stats_manager,
            gpu_stats_manager,
        }
    }
}


pub struct CPUStat {
    pub info: String,
    pub count: usize,
}
pub struct CPUStatsManager {
    system: System,
}

impl CPUStatsManager {
    pub fn init() -> Self {
        Self { system: System::new_all() }
    }

    pub fn get_stats(self: &mut Self) -> CPUStat {
        self.system.refresh_cpu();
        let cpus = self.system.cpus();
        let count = cpus.len();
        let info = if count > 0 {
            let cpu = &cpus[0];
            cpu.brand().to_string()
        } else {
            "unknown".to_string()
        };

        CPUStat {
            info,
            count,
        }
    }
}

pub struct GPUStatsManager {
    // In compilation envs where there is no cuda,
    // We do not use the nvml, therefore setting it as None.
    nvml: Option<Nvml>,
}

impl GPUStatsManager {
    pub fn init() -> Self {
        // In cases of MacOS or docker containers where --gpus are not specified,
        // the Nvml::init() would return an error. In there scenarios, we
        // assign Self::nvml to be None, indicating that the current runtime
        // environment does not support cuda interface.
        let nvml_res = Nvml::init();
        let nvml = match nvml_res {
            Ok(n) => Some(n),
            Err(_) => None,
        };
        Self {
            nvml,
        }
    }

    pub fn get_stats(self: &Self) -> Result<Vec<String>> {
        let mut res = vec![];
        if self.nvml.is_none() {
            // If cuda is not supported in the runtime environment,
            // just return an empty list.
            return Ok(res);
        }

        let nvml = self.nvml.as_ref().unwrap();
        let device_count = nvml.device_count()?;
        for i in 0..device_count {
            let name = nvml.device_by_index(i)?.name()?;
            res.push(name);
        }
        Ok(res)
    }
}