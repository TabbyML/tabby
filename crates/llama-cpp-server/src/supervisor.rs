use std::{env::var, net::TcpListener, process::Stdio, time::Duration};

use tokio::{io::AsyncBufReadExt, task::JoinHandle};
use tracing::{debug, warn};
use which::which;

use crate::api_endpoint;

pub struct LlamaCppSupervisor {
    name: &'static str,
    port: u16,
    handle: JoinHandle<()>,
}

impl LlamaCppSupervisor {
    pub fn new(
        name: &'static str,
        num_gpu_layers: u16,
        embedding: bool,
        model_path: &str,
        parallelism: u8,
        chat_template: Option<String>,
        enable_fast_attention: bool,
        context_size: usize,
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
                    .to_string();
                let mut command = tokio::process::Command::new(server_binary);

                command
                    .arg("-m")
                    .arg(&model_path)
                    .arg("--cont-batching")
                    .arg("--port")
                    .arg(port.to_string())
                    .arg("-np")
                    .arg(parallelism.to_string())
                    .arg("--log-disable")
                    .arg("--ctx-size")
                    .arg(context_size.to_string())
                    .kill_on_drop(true)
                    .stderr(Stdio::piped())
                    .stdout(Stdio::null());

                if let Ok(n_threads) = std::env::var("LLAMA_CPP_N_THREADS") {
                    command.arg("-t").arg(n_threads);
                }

                if num_gpu_layers > 0 {
                    command.arg("-ngl").arg(num_gpu_layers.to_string());
                }

                if embedding {
                    command
                        .arg("--embedding")
                        .arg("--ubatch-size")
                        .arg(var("LLAMA_CPP_EMBEDDING_N_UBATCH_SIZE").unwrap_or("4096".into()));
                }

                if let Some(chat_template) = chat_template.as_ref() {
                    command.arg("--chat-template").arg(chat_template);
                }

                if enable_fast_attention {
                    command.arg("-fa");
                };

                let command_args = format!("{:?}", command);

                let mut process = command.spawn().unwrap_or_else(|e| {
                    panic!(
                        "Failed to start llama-server <{}> with command {:?}: {}",
                        name, command, e
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
                        "llama-server <{}> exited with status code {}, args: `{}`",
                        name, status_code, command_args
                    );
                    let mut stderr = process
                        .stderr
                        .take()
                        .map(tokio::io::BufReader::new)
                        .map(|reader| reader.lines())
                        .expect("Failed to read stderr");

                    while let Ok(Some(line)) = stderr.next_line().await {
                        warn!("<{}>: {}", name, line);
                    }

                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        });

        Self { name, handle, port }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub async fn start(&self) {
        debug!("Waiting for llama-server <{}> to start...", self.name);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        loop {
            let Ok(resp) = client
                .get(api_endpoint(self.port) + "/health")
                .timeout(Duration::from_secs(1))
                .send()
                .await
            else {
                debug!("llama-server <{}> not ready yet, retrying...", self.name);
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            };

            if resp.status().is_success() {
                debug!("llama-server <{}> started successfully", self.name);
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
