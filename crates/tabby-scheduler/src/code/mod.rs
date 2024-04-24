use tabby_common::{Point, Tag};
use text_splitter::{Characters, TextSplitter};
use tree_sitter_tags::TagsContext;

mod languages;

pub struct CodeIntelligence {
    context: TagsContext,
    splitter: TextSplitter<Characters>,
}

impl Default for CodeIntelligence {
    fn default() -> Self {
        Self {
            context: TagsContext::new(),
            splitter: TextSplitter::default().with_trim_chunks(true),
        }
    }
}

impl CodeIntelligence {
    pub fn find_tags(&mut self, language: &str, content: &str) -> Vec<Tag> {
        let config = languages::get(language);
        let empty = Vec::new();

        let Some(config) = config else {
            return empty;
        };

        let Ok((tags, has_error)) = self
            .context
            .generate_tags(&config.0, content.as_bytes(), None)
        else {
            return empty;
        };

        if has_error {
            return empty;
        }

        tags.filter_map(|x| x.ok())
            .map(|x| Tag {
                range: x.range,
                name_range: x.name_range,
                utf16_column_range: x.utf16_column_range,
                line_range: x.line_range,
                docs: x.docs,
                is_definition: x.is_definition,
                syntax_type_name: config.0.syntax_type_name(x.syntax_type_id).to_owned(),
                span: Point::new(x.span.start.row, x.span.start.column)
                    ..Point::new(x.span.end.row, x.span.end.column),
            })
            .collect()
    }

    // FIXME(meng): implement with treesitter based CodeSplitter.
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (i64, &'text str)> + 'splitter {
        self.splitter.chunk_indices(text, 256).map(|(offset, chunk)| {
            (line_number_from_byte_offset(text, offset), chunk)
        })
    }
}

fn line_number_from_byte_offset(s: &str, byte_offset: usize) -> i64{
    let mut line_number = 1; // Start counting from line 1
    let mut current_offset = 0;

    for c in s.chars() {
        if c == '\n' {
            line_number += 1;
        }
        current_offset += c.len_utf8();
        if current_offset >= byte_offset {
            break;
        }
    }

    line_number
}
