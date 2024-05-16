use std::{env, net::TcpListener, process::Stdio};

use tokio::task::JoinHandle;
use tracing::warn;
use which::which;

use crate::api_endpoint;

pub struct LlamaCppSupervisor {
    port: u16,
    handle: JoinHandle<()>,
}

impl LlamaCppSupervisor {
    pub fn new(
        use_gpu: bool,
        embedding: bool,
        model_path: &str,
        parallelism: u8,
    ) -> LlamaCppSupervisor {
        let Some(binary_name) = find_binary_name() else {
            panic!("Failed to locate llama-server binary, please make sure you have llama-server binary locates in the same directory as the current executable.");
        };

        let model_path = model_path.to_owned();
        let port = get_available_port();
        let handle = tokio::spawn(async move {
            loop {
                let server_binary = std::env::current_exe()
                    .expect("Failed to get current executable path")
                    .parent()
                    .expect("Failed to get parent directory")
                    .join(&binary_name)
                    .display()
                    .to_string()
                    + std::env::consts::EXE_SUFFIX;
                let mut command = tokio::process::Command::new(server_binary);

                command
                    .arg("-m")
                    .arg(&model_path)
                    .arg("--port")
                    .arg(port.to_string())
                    .arg("-np")
                    .arg(parallelism.to_string())
                    .arg("--log-disable")
                    .arg("--ctx-size")
                    .arg(env::var("LLAMA_CPP_N_CONTEXT_SIZE").unwrap_or("4096".into()))
                    .arg("--batch-size")
                    .arg(env::var("LLAMA_CPP_N_BATCH_SIZE").unwrap_or("512".into()))
                    .kill_on_drop(true)
                    .stderr(Stdio::null())
                    .stdout(Stdio::null());

                if let Ok(n_threads) = std::env::var("LLAMA_CPP_N_THREADS") {
                    command.arg("-t").arg(n_threads);
                }

                if use_gpu {
                    let num_gpu_layers =
                        std::env::var("LLAMA_CPP_N_GPU_LAYERS").unwrap_or("9999".into());
                    command.arg("-ngl").arg(&num_gpu_layers);
                }

                if embedding {
                    command.arg("--embedding");
                }

                let mut process = command.spawn().unwrap_or_else(|e| {
                    panic!(
                        "Failed to start llama-server with command {:?}: {}",
                        command, e
                    )
                });

                let status_code = process
                    .wait()
                    .await
                    .ok()
                    .and_then(|s| s.code())
                    .unwrap_or(-1);

                if status_code != 0 {
                    warn!(
                        "llama-server exited with status code {}, restarting...",
                        status_code
                    );
                }
            }
        });

        Self { handle, port }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub async fn start(&self) {
        let client = reqwest::Client::new();
        loop {
            let Ok(resp) = client.get(api_endpoint(self.port) + "/health").send().await else {
                continue;
            };

            if resp.status().is_success() {
                return;
            }
        }
    }
}

fn find_binary_name() -> Option<String> {
    let current_exe = std::env::current_exe().expect("Failed to get current executable path");
    let binary_dir = current_exe
        .parent()
        .expect("Failed to get parent directory");
    let binary_name = "llama-server".to_owned();
    let binary_from_path = which("llama-server")
        .ok()
        .map(|path| path.display().to_string());
    std::fs::read_dir(binary_dir)
        .expect("Failed to read directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with(&binary_name)
        })
        .map(|entry| entry.path().display().to_string())
        .next()
        .or(binary_from_path)
}

fn get_available_port() -> u16 {
    (30888..40000)
        .find(|port| port_is_available(*port))
        .expect("Failed to find available port")
}

fn port_is_available(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

impl Drop for LlamaCppSupervisor {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
