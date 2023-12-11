use std::path::PathBuf;

use insta::assert_yaml_snapshot;
use lazy_static::lazy_static;
use serde_json::json;
use tokio::{
    process::Command,
    time::{sleep, Duration},
};

lazy_static! {
    static ref SERVER: bool = {
        let mut cmd = Command::new(tabby_path());
        cmd.arg("serve")
            .arg("--model")
            .arg("TabbyML/StarCoder-1B")
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

async fn golden_test(body: serde_json::Value) -> serde_json::Value {
    let mut body = body.clone();
    body.as_object_mut().unwrap().insert(
        "debug_options".to_owned(),
        json!({
            "disable_retrieval_augmented_code_completion": true
        }),
    );

    let actual: serde_json::Value = CLIENT
        .post("http://localhost:9090/v1/completions")
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    actual
}

async fn assert_golden(body: serde_json::Value) {
    assert_yaml_snapshot!(golden_test(body).await, {
        ".id" => "test-id"
    });
}

#[tokio::test]
async fn run_golden_tests() {
    wait_for_server().await;

    assert_golden(json!({
            "language": "python",
            "segments": {
                "prefix": "def fib(n):\n    ",
                "suffix": "\n        return fib(n - 1) + fib(n - 2)"
            }
    }))
    .await;

    assert_golden(json!({
            "language": "python",
            "segments": {
                "prefix": "import datetime\n\ndef parse_expenses(expenses_string):\n    \"\"\"Parse the list of expenses and return the list of triples (date, value, currency).\n    Ignore lines starting with #.\n    Parse the date using datetime.\n    Example expenses_string:\n        2016-01-02 -34.01 USD\n        2016-01-03 2.59 DKK\n        2016-01-03 -2.72 EUR\n    \"\"\"\n    for line in expenses_string.split('\\n'):\n        "
            }
    })).await;
}
