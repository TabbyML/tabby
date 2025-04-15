use super::{EditHistory, EditRange};

pub struct NextEditPromptBuilder;

impl NextEditPromptBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build_prompt(&self, edit_history: &EditHistory) -> String {
        let prompt = format!("<|original_code|>\n{}\n<|edits_diff|>\n{}\n<|current_version|>\n{}\n<|next_version|>\n",
            edit_history.original_code,
            edit_history.edits_diff,
            edit_history.current_version.content
        );

        prompt
    }

    pub fn parse_edit_range(
        &self,
        text: &str,
        cursor_line: u32,
        cursor_character: u32,
    ) -> EditRange {
        // For this basic implementation, we'll assume the edit happens at cursor position
        // and extends to the end of the predicted text. In a real implementation, you would
        // need more sophisticated parsing based on the model's output.

        let lines: Vec<&str> = text.lines().collect();
        let line_count = lines.len() as u32;

        let last_line = if line_count > 0 {
            lines.last().unwrap()
        } else {
            ""
        };

        EditRange {
            start_line: cursor_line,
            start_character: cursor_character,
            end_line: cursor_line + line_count.saturating_sub(1),
            end_character: if line_count <= 1 {
                cursor_character + last_line.len() as u32
            } else {
                last_line.len() as u32
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::completion::{CurrentVersion, CursorPosition};

    #[test]
    fn test_build_prompt() {
        let edit_history = EditHistory {
            original_code: "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            edits_diff: "---src/main.rs\n+++src/main.rs\n@@ -1,1 +1,2 @@\n    println!(\"Hello, world!\");\n    let x = 5;\n    println!(\"Hello, world!\");".to_string(),
            current_version: CurrentVersion {
                content: "fn main() {\n    let x = 5;\n    println!(\"Hello, world!\");\n}".to_string(),
                cursor_position: CursorPosition {
                    line: 2,
                    character: 0,
                },
            },
        };

        let builder = NextEditPromptBuilder::new();
        let prompt = builder.build_prompt(&edit_history);

        // Check that prompt contains all the important parts
        assert!(prompt.contains("<|original_code|>"));
        assert!(prompt.contains("<|edits_diff|>"));
        assert!(prompt.contains("<|current_version|>"));
        assert!(prompt.contains("fn main()"));
        assert!(prompt.contains("let x = 5;"));
    }

    #[test]
    fn test_parse_edit_range() {
        let builder = NextEditPromptBuilder::new();

        // Test single line edit
        let text = "    println!(\"The value of x is: {}\", x);";
        let range = builder.parse_edit_range(text, 2, 4);
        assert_eq!(range.start_line, 2);
        assert_eq!(range.start_character, 4);
        assert_eq!(range.end_line, 2);
        assert_eq!(range.end_character, 4 + text.len() as u32);

        // Test multi-line edit
        let text = "    let y = 10;\n    println!(\"The value of y is: {}\", y);";
        let range = builder.parse_edit_range(text, 2, 4);
        assert_eq!(range.start_line, 2);
        assert_eq!(range.start_character, 4);
        assert_eq!(range.end_line, 3);
        assert_eq!(
            range.end_character,
            "    println!(\"The value of y is: {}\", y);".len() as u32
        );
    }
}
