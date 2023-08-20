use anyhow::Result;
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

    pub fn get_gpu_stats(self: &Self) -> Result<Vec<String>> {
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