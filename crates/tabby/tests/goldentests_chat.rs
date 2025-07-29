use std::path::PathBuf;

use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;
use serde_json::json;
use serial_test::serial;
use tokio::{
    process::Command,
    time::{sleep, Duration},
};

#[derive(Deserialize)]
pub struct ChatCompletionChunk {
    choices: [ChatCompletionChoice; 1],
}

#[derive(Deserialize)]
pub struct ChatCompletionChoice {
    delta: ChatCompletionDelta,
}

#[derive(Deserialize)]
pub struct ChatCompletionDelta {
    content: Option<String>,
}

lazy_static! {
    static ref CLIENT: reqwest::Client = reqwest::Client::new();
}

fn workspace_dir() -> PathBuf {
    let output = std::process::Command::new(env!("CARGO"))
        .arg("locate-project")
        .arg("--workspace")
        .arg("--message-format=plain")
        .output()
        .unwrap()
        .stdout;
    let cargo_path = std::path::Path::new(std::str::from_utf8(&output).unwrap().trim());
    cargo_path.parent().unwrap().to_path_buf()
}

fn tabby_path() -> PathBuf {
    workspace_dir().join("target/debug/tabby")
}

fn initialize_server(gpu_device: Option<&str>) {
    let mut cmd = Command::new(tabby_path());
    cmd.arg("serve")
        .arg("--chat-model")
        .arg("TabbyML/Mistral-7B")
        .arg("--no-webserver")
        .arg("--port")
        .arg("9090")
        .kill_on_drop(true);

    if let Some(gpu_device) = gpu_device {
        cmd.arg("--device").arg(gpu_device);
    }

    tokio::task::spawn(async move {
        cmd.spawn()
            .expect("Failed to start server")
            .wait()
            .await
            .unwrap();
    });
}

async fn wait_for_server(gpu_device: Option<&str>) {
    initialize_server(gpu_device);

    loop {
        println!("Waiting for server to start...");
        match reqwest::get("http://127.0.0.1:9090/v1/health").await {
            Ok(resp) => {
                if resp.status().is_success() {
                    break;
                }
            }
            Err(e) => {
                println!("Waiting for server to start: {e:?}");
            }
        }
        sleep(Duration::from_secs(5)).await;
    }
}

async fn golden_test(body: serde_json::Value) -> String {
    let mut es = EventSource::new(
        CLIENT
            .post("http://127.0.0.1:9090/v1/chat/completions")
            .json(&body),
    )
    .unwrap();

    let mut actual = "".to_owned();
    while let Some(event) = es.next().await {
        match event {
            Ok(Event::Open) => {}
            Ok(Event::Message(message)) => {
                let x: ChatCompletionChunk = serde_json::from_str(&message.data).unwrap();
                if let Some(content) = &x.choices[0].delta.content {
                    actual += content
                }
            }
            Err(e) => {
                match e {
                    reqwest_eventsource::Error::StreamEnded => {
                        break;
                    }
                    reqwest_eventsource::Error::InvalidStatusCode(code, resp) => {
                        let resp = resp.text().await.unwrap();
                        println!("Error: {code} {resp:?}");
                    }
                    e => {
                        println!("Error: {e:?}");
                    }
                }
                break;
            }
        }
    }

    actual
}

macro_rules! assert_golden {
    ($expr:expr) => {
        insta::assert_yaml_snapshot!(golden_test($expr).await);
    };
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
#[serial]
async fn run_chat_golden_tests() {
    wait_for_server(Some("metal")).await;

    assert_golden!(json!({
            "seed": 0,
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": "How to convert a list of string to numbers in python"
                }
            ]
    }));

    assert_golden!(json!({
            "seed": 0,
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": "How to parse email address with regex"
                }
            ]
    }));
}

#[tokio::test]
#[serial]
async fn run_chat_golden_tests_cpu() {
    wait_for_server(Some("cpu")).await;

    assert_golden!(json!({
            "seed": 0,
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": "How are you?"
                }
            ]
    }));
}
