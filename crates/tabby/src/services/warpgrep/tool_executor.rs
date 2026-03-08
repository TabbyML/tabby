use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use tracing::debug;

pub struct ToolCall {
    pub name: String,
    pub params: HashMap<String, String>,
}

/// Validate that a path does not escape the repo directory via traversal.
fn validate_path(repo_dir: &Path, relative: &str) -> Option<std::path::PathBuf> {
    let joined = repo_dir.join(relative);
    // Canonicalize both to resolve symlinks and ".."
    let canonical = joined.canonicalize().ok()?;
    let base = repo_dir.canonicalize().ok()?;
    if canonical.starts_with(&base) {
        Some(canonical)
    } else {
        None
    }
}

pub fn execute_ripgrep(repo_dir: &Path, pattern: &str, path: Option<&str>) -> String {
    // Validate path if provided
    let search_dir = if let Some(p) = path {
        match validate_path(repo_dir, p) {
            Some(validated) => validated,
            None => return format!("Invalid path: {p}"),
        }
    } else {
        repo_dir.to_path_buf()
    };

    let mut cmd = Command::new("grep");
    // Use -F (fixed string) instead of -E (regex) to prevent ReDoS from LLM-supplied patterns
    cmd.arg("-rn").arg("-F").arg(pattern);
    cmd.arg(&search_dir);

    match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.is_empty() {
                return "No matches found.".to_string();
            }
            // Strip the repo_dir prefix from output paths
            let prefix = repo_dir.to_string_lossy();
            let cleaned: String = stdout
                .lines()
                .take(100)
                .map(|line| {
                    line.strip_prefix(prefix.as_ref())
                        .and_then(|s| s.strip_prefix('/'))
                        .unwrap_or(line)
                })
                .collect::<Vec<_>>()
                .join("\n");
            truncate_string(cleaned, 4000)
        }
        Err(e) => {
            debug!("grep error: {}", e);
            format!("Error: {e}")
        }
    }
}

pub fn execute_read(repo_dir: &Path, path: &str, lines: Option<&str>) -> String {
    let file_path = match validate_path(repo_dir, path) {
        Some(p) => p,
        None => return format!("Invalid path: {path}"),
    };
    match fs::read_to_string(&file_path) {
        Ok(content) => {
            if let Some(line_spec) = lines {
                extract_line_ranges(&content, line_spec)
            } else {
                truncate_string(content, 8000)
            }
        }
        Err(_) => format!("File not found: {path}"),
    }
}

pub fn execute_list_directory(repo_dir: &Path, path: &str) -> String {
    let dir_path = if path.is_empty() {
        repo_dir.to_path_buf()
    } else {
        match validate_path(repo_dir, path) {
            Some(p) => p,
            None => return format!("Invalid path: {path}"),
        }
    };

    match fs::read_dir(&dir_path) {
        Ok(entries) => {
            let mut items: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        format!("{name}/")
                    } else {
                        name
                    }
                })
                .collect();
            items.sort();
            truncate_string(items.join("\n"), 4000)
        }
        Err(_) => format!("Directory not found: {path}"),
    }
}

pub fn parse_tool_calls(content: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let tool_names = ["ripgrep", "read", "list_directory", "finish"];

    for tool_name in &tool_names {
        let open_tag = format!("<{tool_name}>");
        let close_tag = format!("</{tool_name}>");

        let mut search_from = 0;
        while let Some(start) = content[search_from..].find(&open_tag) {
            let abs_start = search_from + start + open_tag.len();
            if let Some(end) = content[abs_start..].find(&close_tag) {
                let inner = &content[abs_start..abs_start + end];
                let params = parse_xml_params(inner);
                calls.push(ToolCall {
                    name: tool_name.to_string(),
                    params,
                });
                search_from = abs_start + end + close_tag.len();
            } else {
                break;
            }
        }
    }

    calls
}

fn parse_xml_params(inner: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    let param_names = ["pattern", "path", "glob", "lines", "result"];

    for param_name in &param_names {
        let open = format!("<{param_name}>");
        let close = format!("</{param_name}>");

        if let Some(start) = inner.find(&open) {
            let val_start = start + open.len();
            if let Some(end) = inner[val_start..].find(&close) {
                let value = inner[val_start..val_start + end].trim().to_string();
                params.insert(param_name.to_string(), value);
            }
        }
    }

    params
}

pub fn format_tool_responses(results: &[(String, String)]) -> String {
    let mut output = String::new();
    for (name, result) in results {
        output.push_str(&format!(
            "<tool_response>\n<{name}>\n{result}\n</{name}>\n</tool_response>\n"
        ));
    }
    output
}

pub fn parse_finish_output(content: &str) -> Vec<(String, Vec<(usize, usize)>)> {
    let mut results = Vec::new();

    // Find <finish> tag and extract its <result> content
    let finish_content = if let Some(start) = content.find("<finish>") {
        let after = &content[start + 8..];
        if let Some(end) = after.find("</finish>") {
            &after[..end]
        } else {
            after
        }
    } else {
        content
    };

    // Extract result tag content
    let result_content = if let Some(start) = finish_content.find("<result>") {
        let after = &finish_content[start + 8..];
        if let Some(end) = after.find("</result>") {
            &after[..end]
        } else {
            after
        }
    } else {
        finish_content
    };

    for line in result_content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Format: path:start-end,start-end
        if let Some((path, ranges_str)) = line.rsplit_once(':') {
            let mut ranges = Vec::new();
            for range in ranges_str.split(',') {
                let range = range.trim();
                if let Some((start_str, end_str)) = range.split_once('-') {
                    if let (Ok(start), Ok(end)) = (
                        start_str.trim().parse::<usize>(),
                        end_str.trim().parse::<usize>(),
                    ) {
                        ranges.push((start, end));
                    }
                }
            }
            if !ranges.is_empty() {
                results.push((path.to_string(), ranges));
            }
        }
    }

    results
}

fn extract_line_ranges(content: &str, line_spec: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut output = String::new();

    for range in line_spec.split(',') {
        let range = range.trim();
        if let Some((start_str, end_str)) = range.split_once('-') {
            if let (Ok(start), Ok(end)) = (
                start_str.trim().parse::<usize>(),
                end_str.trim().parse::<usize>(),
            ) {
                let start = start.saturating_sub(1); // 1-indexed to 0-indexed
                let end = end.min(lines.len());
                for (i, line) in lines[start..end].iter().enumerate() {
                    output.push_str(&format!("{}:{}\n", start + i + 1, line));
                }
            }
        }
    }

    truncate_string(output, 8000)
}

fn truncate_string(s: String, max_len: usize) -> String {
    if s.len() <= max_len {
        return s;
    }
    // Find a char boundary at or before max_len to avoid panicking on multi-byte UTF-8
    let boundary = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max_len)
        .last()
        .unwrap_or(0);
    let mut truncated = s[..boundary].to_string();
    truncated.push_str("\n... (truncated)");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls_ripgrep() {
        let content = r#"I'll search for the pattern.
<ripgrep>
<pattern>fn main</pattern>
<path>src</path>
</ripgrep>"#;
        let calls = parse_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ripgrep");
        assert_eq!(calls[0].params.get("pattern").unwrap(), "fn main");
        assert_eq!(calls[0].params.get("path").unwrap(), "src");
    }

    #[test]
    fn test_parse_tool_calls_finish() {
        let content = r#"<finish>
<result>
src/main.rs:1-50,80-100
src/lib.rs:10-30
</result>
</finish>"#;
        let calls = parse_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "finish");
    }

    #[test]
    fn test_parse_finish_output() {
        let content = r#"<finish>
<result>
src/main.rs:1-50,80-100
src/lib.rs:10-30
</result>
</finish>"#;
        let results = parse_finish_output(content);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "src/main.rs");
        assert_eq!(results[0].1, vec![(1, 50), (80, 100)]);
        assert_eq!(results[1].0, "src/lib.rs");
        assert_eq!(results[1].1, vec![(10, 30)]);
    }

    #[test]
    fn test_parse_tool_calls_multiple() {
        let content = r#"<ripgrep>
<pattern>struct Config</pattern>
</ripgrep>
<read>
<path>src/config.rs</path>
<lines>1-50</lines>
</read>"#;
        let calls = parse_tool_calls(content);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "ripgrep");
        assert_eq!(calls[1].name, "read");
        assert_eq!(calls[1].params.get("lines").unwrap(), "1-50");
    }

    #[test]
    fn test_format_tool_responses() {
        let results = vec![(
            "ripgrep".to_string(),
            "src/main.rs:1:fn main() {}".to_string(),
        )];
        let output = format_tool_responses(&results);
        assert!(output.contains("<tool_response>"));
        assert!(output.contains("<ripgrep>"));
        assert!(output.contains("src/main.rs:1:fn main() {}"));
    }

    #[test]
    fn test_extract_line_ranges() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let result = extract_line_ranges(content, "2-4");
        assert!(result.contains("2:line2"));
        assert!(result.contains("3:line3"));
        assert!(result.contains("4:line4"));
        assert!(!result.contains("1:line1"));
    }
}
