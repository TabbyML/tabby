use tantivy::schema::{Field, Schema, STORED, STRING};

pub struct DocSearchSchema {
    pub schema: Schema,

    // === Fields for document ===
    pub field_id: Field,
    pub field_title: Field,
    pub field_link: Field,
    pub field_body: Field,

    // === Fields for chunk ===
    pub field_chunk_id: Field,
    // Binarized embedding tokens with the following mapping:
    // * [-1, 0] -> 0
    // * (0, 1] -> 1
    pub field_chunk_embedding_token: Field,
}

impl DocSearchSchema {
    pub fn new() -> Self {
        let mut builder = Schema::builder();

        let field_id = builder.add_text_field("id", STRING | STORED);
        let field_title = builder.add_text_field("title", STORED);
        let field_link = builder.add_text_field("link", STORED);
        let field_body = builder.add_text_field("body", STORED);

        let field_chunk_id = builder.add_text_field("chunk_id", STRING | STORED);
        let field_chunk_embedding_token = builder.add_text_field("chunk_embedding_token", STRING);

        let schema = builder.build();

        Self {
            schema,
            field_id,
            field_title,
            field_link,
            field_body,

            field_chunk_id,
            field_chunk_embedding_token,
        }
    }

    pub fn binarize_embedding<'a>(
        embedding: impl Iterator<Item = &'a f32> + 'a,
    ) -> impl Iterator<Item = String> + 'a {
        embedding.enumerate().map(|(i, value)| {
            if *value <= 0.0 {
                format!("embedding_zero_{}", i)
            } else {
                format!("embedding_one_{}", i)
            }
        })
    }
}

impl Default for DocSearchSchema {
    fn default() -> Self {
        Self::new()
    }
}
