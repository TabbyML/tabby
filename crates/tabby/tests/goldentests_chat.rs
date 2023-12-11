use std::path::PathBuf;

use insta::assert_yaml_snapshot;
use lazy_static::lazy_static;
use serde::Deserialize;
use serde_json::json;
use serde_jsonlines::BufReadExt;
use tokio::{
    process::Command,
    time::{sleep, Duration},
};

#[derive(Deserialize)]
pub struct ChatCompletionChunk {
    content: String,
}

lazy_static! {
    static ref SERVER: bool = {
        let mut cmd = Command::new(tabby_path());
        cmd.arg("serve")
            .arg("--chat-model")
            .arg("TabbyML/Mistral-7B")
            .arg("--port")
            .arg("9090")
            .arg("--device")
            .arg("metal")
            .kill_on_drop(true);
        tokio::task::spawn(async move {
            cmd.spawn()
                .expect("Failed to start server")
                .wait()
                .await
                .unwrap();
        });
        true
    };
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

async fn wait_for_server() {
    lazy_static::initialize(&SERVER);

    loop {
        println!("Waiting for server to start...");
        let is_ok = reqwest::get("http://localhost:9090/v1/health")
            .await
            .is_ok();
        if is_ok {
            break;
        } else {
            sleep(Duration::from_secs(5)).await;
        }
    }
}

async fn golden_test(body: serde_json::Value) -> String {
    let bytes = CLIENT
        .post("http://localhost:9090/v1beta/chat/completions")
        .json(&body)
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let lines = bytes.json_lines::<ChatCompletionChunk>();
    let mut actual = "".to_owned();
    for x in lines {
        actual += &x.unwrap().content;
    }

    actual
}

async fn assert_golden(body: serde_json::Value) {
    assert_yaml_snapshot!(golden_test(body).await);
}

#[tokio::test]
async fn run_chat_golden_tests() {
    wait_for_server().await;

    assert_golden(json!({
            "messages": [
                {
                    "role": "user",
                    "content": "How to convert a list of string to numbers in python"
                }
            ]
    }))
    .await;

    assert_golden(json!({
            "messages": [
                {
                    "role": "user",
                    "content": "How to parse email address with regex"
                }
            ]
    }))
    .await;
}
