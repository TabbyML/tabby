use std::{cmp::min, io::Error};

use futures_util::StreamExt;
use regex::Regex;
use reqwest::Client;
use tokio_util::io::ReaderStream;

use crate::{
    address::ParsedAddress, bar::WrappedBar, consts::*, error::ValidateError, hash::HashChecker, io,
};

pub struct HTTPSHandler;
impl HTTPSHandler {
    pub async fn get(
        input: &str,
        output: &str,
        bar: &mut WrappedBar,
        expected_sha256: &str,
    ) -> Result<(), ValidateError> {
        HTTPSHandler::_get(input, output, bar).await?;
        HashChecker::check(output, expected_sha256)
    }

    pub async fn put(input: &str, output: &str, mut bar: WrappedBar) -> Result<(), ValidateError> {
        let parsed_address = ParsedAddress::parse_address(output, bar.silent);
        let file = tokio::fs::File::open(&input)
            .await
            .expect("Cannot open input file for HTTPS read");
        let total_size = file
            .metadata()
            .await
            .expect("Cannot determine input file size for HTTPS read")
            .len();
        let input_ = input.to_string();
        let output_ = output.to_string();
        let mut reader_stream = ReaderStream::new(file);

        let mut uploaded = HTTPSHandler::get_already_uploaded(output, bar.silent).await;
        bar.set_length(total_size);

        let async_stream = async_stream::stream! {
            while let Some(chunk) = reader_stream.next().await {
                if let Ok(chunk) = &chunk {
                    let new = min(uploaded + (chunk.len() as u64), total_size);
                    uploaded = new;
                    bar.set_position(new);
                    if uploaded >= total_size {
                        bar.finish_upload(&input_, &output_);
                    }
                }
                yield chunk;
            }
        };

        let response = reqwest::Client::new()
            .put(output)
            .header("content-type", "application/octet-stream")
            .header(
                "Range",
                "bytes=".to_owned() + &uploaded.to_string()[..] + "-",
            )
            .header(
                reqwest::header::USER_AGENT,
                reqwest::header::HeaderValue::from_static(CLIENT_ID),
            )
            .basic_auth(parsed_address.username, Some(parsed_address.password))
            .body(reqwest::Body::wrap_stream(async_stream))
            .send()
            .await
            .unwrap();
        println!("{:?}", response.text().await.unwrap());
        Ok(())
    }

    pub async fn get_links(input: String) -> Result<Vec<String>, Error> {
        let mut result = Vec::new();
        let res = HTTPSHandler::list(&input).await.unwrap();
        let lines: Vec<&str> = res.split('\n').collect();

        for line in lines {
            let re = Regex::new(r#".*href="/(.+?)".*"#).unwrap();
            let caps = re.captures(line);
            if let Some(e) = caps {
                result.push(e.get(1).unwrap().as_str().to_string())
            }
        }
        result.push("..".to_string());

        result.sort();

        Ok(result)
    }

    async fn list(input: &str) -> Result<String, ValidateError> {
        let is_silent = true;
        let parsed_address = ParsedAddress::parse_address(input, is_silent);

        let res = Client::new()
            .get(input)
            .header(
                reqwest::header::USER_AGENT,
                reqwest::header::HeaderValue::from_static(CLIENT_ID),
            )
            .basic_auth(parsed_address.username, Some(parsed_address.password))
            .send()
            .await
            .map_err(|_| format!("Failed to GET from {}", &input))
            .unwrap()
            .text()
            .await
            .unwrap();

        Ok(res)
    }

    async fn _get(input: &str, output: &str, bar: &mut WrappedBar) -> Result<(), ValidateError> {
        let parsed_address = ParsedAddress::parse_address(input, bar.silent);
        let (mut out, mut downloaded) = io::get_output(output, bar.silent);

        let res = Client::new()
            .get(input)
            .header(
                "Range",
                "bytes=".to_owned() + &downloaded.to_string()[..] + "-",
            )
            .header(
                reqwest::header::USER_AGENT,
                reqwest::header::HeaderValue::from_static(CLIENT_ID),
            )
            .basic_auth(parsed_address.username, Some(parsed_address.password))
            .send()
            .await
            .map_err(|_| format!("Failed to GET from {} to {}", &input, &output))
            .unwrap();
        let total_size = downloaded + res.content_length().unwrap_or(0);

        bar.set_length(total_size);

        let mut stream = res.bytes_stream();
        while let Some(item) = stream.next().await {
            let chunk = item.map_err(|_| "Error while downloading.").unwrap();
            out.write_all(&chunk)
                .map_err(|_| "Error while writing to output.")
                .unwrap();
            let new = min(downloaded + (chunk.len() as u64), total_size);
            downloaded = new;
            bar.set_position(new);
        }

        bar.finish_download(input, output);
        Ok(())
    }

    async fn get_already_uploaded(output: &str, silent: bool) -> u64 {
        let parsed_address = ParsedAddress::parse_address(output, silent);
        let res = Client::new()
            .get(output)
            .header(
                reqwest::header::USER_AGENT,
                reqwest::header::HeaderValue::from_static(CLIENT_ID),
            )
            .basic_auth(parsed_address.username, Some(parsed_address.password))
            .send()
            .await
            .map_err(|_| format!("Failed to GET already uploaded size from {}", &output))
            .unwrap();
        res.content_length().unwrap_or(0)
    }
}

#[ignore]
#[tokio::test]
async fn get_https_works() {
    let expected_hash = "0e0f0d7139c8c7e3ff20cb243e94bc5993517d88e8be8d59129730607d5c631b";
    let out_file = "tokei-x86_64-unknown-linux-gnu.tar.gz";

    let result = HTTPSHandler::get("https://github.com/XAMPPRocky/tokei/releases/download/v12.0.4/tokei-x86_64-unknown-linux-gnu.tar.gz", out_file, &mut WrappedBar::new_empty(), expected_hash).await;

    assert!(result.is_ok());
    std::fs::remove_file(out_file).unwrap();
}

#[ignore]
#[tokio::test]
async fn get_resume_works() {
    let expected_size = 561553;
    let out_file = "test/dua-v2.10.2-x86_64-unknown-linux-musl.tar.gz";
    std::fs::copy(
        "test/incomplete_dua-v2.10.2-x86_64-unknown-linux-musl.tar.gz",
        out_file,
    )
    .unwrap();

    let _ = HTTPSHandler::get("https://github.com/Byron/dua-cli/releases/download/v2.10.2/dua-v2.10.2-x86_64-unknown-linux-musl.tar.gz", out_file, &mut WrappedBar::new_empty_verbose(), "").await;

    let actual_size = std::fs::metadata(out_file).unwrap().len();
    assert_eq!(actual_size, expected_size);
    std::fs::remove_file(out_file).unwrap();
}

#[ignore]
#[tokio::test]
async fn list_works_when_typical() {
    let expected = r#"<!doctype html>
<html>
<head>
    <title>Example Domain</title>

    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style type="text/css">
    body {
        background-color: #f0f0f2;
        margin: 0;
        padding: 0;
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    div {
        width: 600px;
        margin: 5em auto;
        padding: 2em;
        background-color: #fdfdff;
        border-radius: 0.5em;
        box-shadow: 2px 3px 7px 2px rgba(0,0,0,0.02);
    }
    a:link, a:visited {
        color: #38488f;
        text-decoration: none;
    }
    @media (max-width: 700px) {
        div {
            margin: 0 auto;
            width: auto;
        }
    }
    </style>
</head>

<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>
"#;

    let result = HTTPSHandler::list("https://example.com").await.unwrap();
    let result = str::replace(&result, "    \n    ", "");
    let result = str::replace(&result, "</style>    \n</head>", "</style>\n</head>");

    assert_eq!(result, expected);
}

#[ignore]
#[tokio::test]
async fn get_links_works_when_typical() {
    let expected = "..";

    let result = HTTPSHandler::get_links("https://github.com/mihaigalos/aim/releases".to_string())
        .await
        .unwrap();

    assert_eq!(result[0], expected);
}
