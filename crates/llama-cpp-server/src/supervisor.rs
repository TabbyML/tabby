use std::{
    collections::VecDeque,
    env::var,
    net::TcpListener,
    process::Stdio,
    time::{Duration, Instant},
};

use tokio::{
    io::{AsyncBufReadExt, BufReader},
    task::JoinHandle,
};
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
        let mut retry_count = 0;
        let initial_time = Instant::now();

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

                let mut stderr = BufReader::new(
                    process
                        .stderr
                        .take()
                        .expect("Failed to get llama.cpp stderr"),
                )
                .lines();
                let mut error_lines = VecDeque::with_capacity(100);

                let wait_handle = process.wait();

                while let Ok(Some(line)) = stderr.next_line().await {
                    if !line.contains("GET /health") {
                        if error_lines.len() >= 100 {
                            error_lines.pop_front();
                        }
                        error_lines.push_back(line);
                    }
                }

                let status_code = wait_handle.await.ok().and_then(|s| s.code()).unwrap_or(-1);

                if status_code != 0 {
                    warn!(
                        "llama-server <{}> exited with status code {}, args: `{}`",
                        name, status_code, command_args
                    );

                    // print only the initial round error message.
                    if retry_count == 0 {
                        eprintln!(
                            "{}\n",
                            tabby_common::terminal::HeaderFormat::BoldRed
                                .format("Recent llama-cpp errors:")
                        );
                    }
                    for line in error_lines {
                        // print only the initial round error message.
                        if retry_count == 0 {
                            eprintln!("{}", line);
                        }
                        if let Some(solution) = analyze_error_message(&line) {
                            let solution_lines: Vec<_> = solution.split('\n').collect();
                            let msg = tabby_common::terminal::InfoMessage::new(
                                "ERROR",
                                tabby_common::terminal::HeaderFormat::BoldRed,
                                &solution_lines,
                            );
                            msg.print();
                            break;
                        }
                    }

                    // exit only after the retry loop has been exhausted 5 times and Tabby was initialing for fewer than 1 minute.
                    if retry_count >= 5 && initial_time.elapsed().as_secs() < 60 {
                        eprintln!(
                            "llama-server <{}> encountered a fatal error. Exiting service. Please check the above logs and suggested solutions for details.",
                            name
                        );
                        std::process::exit(1);
                    }

                    retry_count += 1;
                    warn!("Attempting to restart the llama-server...");
                    tokio::time::sleep(Duration::from_secs(1)).await;
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

fn analyze_error_message(error_message: &str) -> Option<String> {
    if error_message.contains("cudaMalloc") {
        return Some(String::from(
            "CUDA memory allocation error detected:\n\
             1. Try using a smaller Model\n\
             2. Try to reduce GPU memory usage\n",
        ));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if error_message.contains("Illegal instruction")
            && !std::arch::is_x86_feature_detected!("avx2")
        {
            return Some(String::from(
                "Illegal instruction detected: Your CPU does not support AVX2 instruction set.\n\
                 Suggestion: Download a compatible binary from https://github.com/ggml-org/llama.cpp/releases"
            ));
        }
    }

    None
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
