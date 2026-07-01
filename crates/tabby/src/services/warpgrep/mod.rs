mod tool_executor;

use std::fs;
use std::path::Path;

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_common::api::code::{
    CodeSearchDocument, CodeSearchError, CodeSearchHit, CodeSearchResponse, CodeSearchScores,
    WarpGrepSearch,
};
use tracing::{debug, warn};

use self::tool_executor::{
    execute_list_directory, execute_read, execute_ripgrep, format_tool_responses,
    parse_finish_output, parse_tool_calls,
};

const DEFAULT_ENDPOINT: &str = "https://api.morphllm.com/v1/chat/completions";
const DEFAULT_MODEL: &str = "morph-warp-grep-v2";
const MAX_TURNS: usize = 4;

pub struct WarpGrepService {
    client: Client,
    api_endpoint: String,
    model: String,
}

pub fn create(api_key: &str) -> impl WarpGrepSearch {
    let endpoint =
        std::env::var("MORPH_API_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string());
    let model = std::env::var("MORPH_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}"))
            .expect("Invalid API key format"),
    );

    let client = Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("Failed to build HTTP client");

    WarpGrepService {
        client,
        api_endpoint: endpoint,
        model,
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

impl WarpGrepService {
    async fn search_impl(&self, repo_dir: &Path, query: &str) -> Result<CodeSearchResponse> {
        let file_tree = build_file_tree(repo_dir)?;
        let initial_msg = format!(
            "<repo_structure>{file_tree}</repo_structure>\n<search_string>{query}</search_string>"
        );

        let mut messages = vec![Message {
            role: "user".to_string(),
            content: initial_msg,
        }];

        for turn in 0..MAX_TURNS {
            let request = ChatRequest {
                model: self.model.clone(),
                messages: messages.clone(),
                temperature: 0.0,
                max_tokens: 2048,
            };

            let resp = self
                .client
                .post(&self.api_endpoint)
                .json(&request)
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("WarpGrep API error {status}: {body}");
            }

            let chat_resp: ChatResponse = resp.json().await?;
            let assistant_msg = chat_resp
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in WarpGrep response"))?
                .message;

            let preview_len = assistant_msg.content.len().min(200);
            debug!(
                "WarpGrep turn {}: {}",
                turn + 1,
                &assistant_msg.content[..preview_len]
            );

            messages.push(assistant_msg.clone());

            let tool_calls = parse_tool_calls(&assistant_msg.content);

            // Check for finish
            if tool_calls.iter().any(|tc| tc.name == "finish") {
                return parse_finish_to_response(repo_dir, &assistant_msg.content);
            }

            if tool_calls.is_empty() {
                debug!("WarpGrep: no tool calls found in response, ending");
                break;
            }

            // Execute tool calls
            let mut results = Vec::new();
            for tc in &tool_calls {
                if tc.name == "finish" {
                    continue;
                }
                let result = match tc.name.as_str() {
                    "ripgrep" => {
                        let pattern = tc.params.get("pattern").map(|s| s.as_str()).unwrap_or("");
                        let path = tc.params.get("path").map(|s| s.as_str());
                        execute_ripgrep(repo_dir, pattern, path)
                    }
                    "read" => {
                        let path = tc.params.get("path").map(|s| s.as_str()).unwrap_or("");
                        let lines = tc.params.get("lines").map(|s| s.as_str());
                        execute_read(repo_dir, path, lines)
                    }
                    "list_directory" => {
                        let path = tc.params.get("path").map(|s| s.as_str()).unwrap_or("");
                        execute_list_directory(repo_dir, path)
                    }
                    _ => format!("Unknown tool: {}", tc.name),
                };
                results.push((tc.name.clone(), result));
            }

            let tool_response = format_tool_responses(&results);
            let turns_used = turn + 1;
            let turns_remaining = MAX_TURNS - turns_used;
            let user_msg = format!(
                "{tool_response}\n[Turns used: {turns_used}/{MAX_TURNS}, Turns remaining: {turns_remaining}]"
            );

            messages.push(Message {
                role: "user".to_string(),
                content: user_msg,
            });
        }

        // If we exhausted turns without finish, return empty
        warn!("WarpGrep exhausted all turns without finish");
        Ok(CodeSearchResponse { hits: vec![] })
    }
}

#[async_trait]
impl WarpGrepSearch for WarpGrepService {
    async fn search(
        &self,
        repo_dir: &Path,
        query: &str,
    ) -> Result<CodeSearchResponse, CodeSearchError> {
        self.search_impl(repo_dir, query)
            .await
            .map_err(CodeSearchError::Other)
    }
}

fn build_file_tree(repo_dir: &Path) -> Result<String> {
    let mut tree = String::new();
    collect_files(repo_dir, repo_dir, &mut tree, 0, 10000)?;
    Ok(tree)
}

fn collect_files(
    base: &Path,
    dir: &Path,
    output: &mut String,
    count: usize,
    limit: usize,
) -> Result<usize> {
    if count >= limit {
        output.push_str("... (file list truncated)\n");
        return Ok(count);
    }

    let mut entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            !name.starts_with('.') && name != "node_modules" && name != "target" && name != "vendor"
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut current_count = count;
    for entry in entries {
        if current_count >= limit {
            output.push_str("... (file list truncated)\n");
            return Ok(current_count);
        }

        let path = entry.path();
        let relative = path
            .strip_prefix(base)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            output.push_str(&relative);
            output.push_str("/\n");
            current_count += 1;
            current_count = collect_files(base, &path, output, current_count, limit)?;
        } else {
            output.push_str(&relative);
            output.push('\n');
            current_count += 1;
        }
    }

    Ok(current_count)
}

fn parse_finish_to_response(repo_dir: &Path, content: &str) -> Result<CodeSearchResponse> {
    let parsed = parse_finish_output(content);
    let mut hits = Vec::new();

    for (filepath, ranges) in parsed {
        for (start, end) in &ranges {
            let body = execute_read(repo_dir, &filepath, Some(&format!("{start}-{end}")));
            let language = detect_language(&filepath);
            let chunk_id = format!("{filepath}:{start}-{end}");

            hits.push(CodeSearchHit {
                scores: CodeSearchScores {
                    rrf: 0.5,
                    bm25: 0.0,
                    embedding: 0.0,
                },
                doc: CodeSearchDocument {
                    file_id: filepath.clone(),
                    chunk_id,
                    body,
                    filepath: filepath.clone(),
                    git_url: String::new(),
                    commit: None,
                    language,
                    start_line: Some(*start),
                },
            });
        }
    }

    Ok(CodeSearchResponse { hits })
}

fn detect_language(filepath: &str) -> String {
    let ext = filepath.rsplit('.').next().unwrap_or("");
    match ext {
        "rs" => "rust",
        "py" => "python",
        "js" => "javascript",
        "ts" => "typescript",
        "tsx" => "typescriptreact",
        "jsx" => "javascriptreact",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "cs" => "csharp",
        "scala" => "scala",
        "sh" | "bash" => "shellscript",
        "lua" => "lua",
        "r" | "R" => "r",
        "sql" => "sql",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "xml" => "xml",
        "html" => "html",
        "css" => "css",
        "md" => "markdown",
        _ => "plaintext",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("src/main.rs"), "rust");
        assert_eq!(detect_language("app.py"), "python");
        assert_eq!(detect_language("index.ts"), "typescript");
        assert_eq!(detect_language("unknown"), "plaintext");
    }
}
