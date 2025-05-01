use super::EditHistory;

pub struct NextEditPromptBuilder;

impl NextEditPromptBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build_prompt(&self, edit_history: &EditHistory) -> String {
        let prompt = format!("<|original_code|>\n{}\n<|edits_diff|>\n{}\n<|current_version|>\n{}\n<|next_version|>\n",
            edit_history.original_code,
            edit_history.edits_diff,
            edit_history.current_version
        );

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let edit_history = EditHistory {
            original_code: "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            edits_diff: "---src/main.rs\n+++src/main.rs\n@@ -1,1 +1,2 @@\n    println!(\"Hello, world!\");\n    let x = 5;\n    println!(\"Hello, world!\");".to_string(),
            current_version: "fn main() {\n    let x = 5;\n    println!(\"Hello, world!\");\n}".to_string(),
        };

        let builder = NextEditPromptBuilder::new();
        let prompt = builder.build_prompt(&edit_history);

        assert!(prompt.contains("<|original_code|>"));
        assert!(prompt.contains("<|edits_diff|>"));
        assert!(prompt.contains("<|current_version|>"));
        assert!(prompt.contains("fn main()"));
        assert!(prompt.contains("let x = 5;"));
    }
}
