use tantivy::schema::{Field, Schema, STORED, STRING};

pub struct DocSearchSchema {
    pub schema: Schema,
    /// Binarized embedding tokens with the following mapping:
    /// * [-1, 0] -> 0
    /// * (0, 1] -> 1
    pub field_embedding_token: Field,

    pub field_title: Field,
    pub field_link: Field,
    pub field_snippet: Field,
}

impl DocSearchSchema {
    pub fn new() -> Self {
        let mut builder = Schema::builder();

        let field_embedding_token = builder.add_text_field("embedding_token", STRING);
        let field_title = builder.add_text_field("title", STORED);
        let field_link = builder.add_text_field("link", STORED);
        let field_snippet = builder.add_text_field("snippet", STORED);

        let schema = builder.build();

        Self {
            schema,
            field_embedding_token,
            field_title,
            field_link,
            field_snippet,
        }
    }
}

impl Default for DocSearchSchema {
    fn default() -> Self {
        Self::new()
    }
}
