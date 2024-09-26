use std::path::PathBuf;

use lazy_static::lazy_static;
use serde_json::json;
use serial_test::serial;
use tokio::{
    process::Command,
    time::{sleep, Duration},
};

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
    let cargo_path = std::path::Path::new(std::str::from_utf8(&output).expect("Valid path").trim());
    cargo_path
        .parent()
        .expect("Path must have a parent folder")
        .to_path_buf()
}

fn tabby_path() -> PathBuf {
    workspace_dir().join("target/debug/tabby")
}

fn initialize_server(gpu_device: Option<&str>) {
    let mut cmd = Command::new(tabby_path());
    cmd.arg("serve")
        .arg("--model")
        .arg("TabbyML/StarCoder-1B")
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
            .expect("Failed to start server");
    });
}

async fn wait_for_server(device: Option<&str>) {
    initialize_server(device);

    loop {
        println!("Waiting for server to start...");
        let is_ok = reqwest::get("http://127.0.0.1:9090/v1/health")
            .await
            .is_ok();
        if is_ok {
            break;
        } else {
            sleep(Duration::from_secs(5)).await;
        }
    }
}

async fn golden_test(body: serde_json::Value) -> serde_json::Value {
    let mut body = body.clone();
    body.as_object_mut().unwrap().insert(
        "debug_options".to_owned(),
        json!({
            "disable_retrieval_augmented_code_completion": true
        }),
    );

    let resp = CLIENT
        .post("http://127.0.0.1:9090/v1/completions")
        .json(&body)
        .send()
        .await
        .unwrap();

    let info = resp.text().await.unwrap();
    eprintln!("info {}", info);

    let actual: serde_json::Value = CLIENT
        .post("http://127.0.0.1:9090/v1/completions")
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    actual
}

macro_rules! assert_golden {
    ($expr:expr) => {
        insta::assert_yaml_snapshot!(golden_test($expr).await, {
            ".id" => "test-id"
        });
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[tokio::test]
#[serial]
async fn run_golden_tests() {
    wait_for_server(Some("metal")).await;

    assert_golden!(json!({
            "language": "python",
            "seed": 0,
            "segments": {
                "prefix": "def fib(n):\n    ",
                "suffix": "\n        return fib(n - 1) + fib(n - 2)"
            }
    }));

    assert_golden!(json!({
            "language": "python",
            "seed": 0,
            "segments": {
                "prefix": "import datetime\n\ndef parse_expenses(expenses_string):\n    \"\"\"Parse the list of expenses and return the list of triples (date, value, currency).\n    Ignore lines starting with #.\n    Parse the date using datetime.\n    Example expenses_string:\n        2016-01-02 -34.01 USD\n        2016-01-03 2.59 DKK\n        2016-01-03 -2.72 EUR\n    \"\"\"\n    for line in expenses_string.split('\\n'):\n        "
            }
    }));
}

#[tokio::test]
#[serial]
async fn run_golden_tests_cpu() {
    wait_for_server(Some("cpu")).await;

    assert_golden!(json!({
            "language": "python",
            "seed": 0,
            "segments": {
                "prefix": "def is_prime(n):\n",
            }
    }));

    assert_golden!(json!({
            "language": "python",
            "seed": 0,
            "segments": {
                "prefix": "def char_frequencies(str):\n  freqs = {}\n  ",
            }
    }));
}
