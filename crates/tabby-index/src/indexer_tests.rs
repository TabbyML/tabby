mod mock_embedding {
    use anyhow::Result;
    use async_trait::async_trait;
    use tabby_inference::Embedding;

    pub struct MockEmbedding {
        result: Vec<f32>,
        error: bool,
    }

    impl MockEmbedding {
        pub fn new(result: Vec<f32>, error: bool) -> Self {
            Self { result, error }
        }
    }

    #[async_trait]
    impl Embedding for MockEmbedding {
        async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
            if self.error {
                Err(anyhow::anyhow!(
                    "Mock error, prompt length {}",
                    prompt.len()
                ))
            } else {
                Ok(self.result.clone())
            }
        }
    }
}

mod structured_doc_tests {
    use std::sync::Arc;

    use serial_test::file_serial;
    use tabby_common::index::{corpus, structured_doc::fields as StructuredDocIndexFields};
    use temp_testdir::TempDir;

    use super::mock_embedding::MockEmbedding;
    use crate::{
        indexer::Indexer,
        public::StructuredDocState,
        structured_doc::public::{
            StructuredDoc, StructuredDocFields, StructuredDocIndexer, StructuredDocIssueFields,
        },
    };

    /// the document should be indexed even no embedding is provided
    /// the document itself could be used for search
    #[test]
    #[file_serial(set_tabby_root)]
    fn test_structured_doc_empty_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let id = "structured_doc_empty_embedding";
        let embedding = MockEmbedding::new(vec![], true);
        let embedding = Arc::new(embedding);
        let indexer = StructuredDocIndexer::new(embedding.clone());
        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: id.to_owned(),
                title: "title".to_owned(),
                author_email: Some("author_email".to_owned()),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let updated_at = chrono::Utc::now();
        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            let updated = indexer
                .presync(&StructuredDocState {
                    id: doc.id().to_string(),
                    updated_at,
                    deleted: false,
                })
                .await
                && indexer.sync(doc).await;
            println!("{updated}");
            updated
        });
        assert!(res);
        indexer.commit();

        let validator = Indexer::new(corpus::STRUCTURED_DOC);

        assert!(validator.is_indexed(id));
        assert!(validator.has_failed_chunks(id));

        tabby_common::path::set_tabby_root(root);
    }

    #[test]
    #[file_serial(set_tabby_root)]
    fn test_structured_doc_with_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let id = "structured_doc_with_embedding";
        let embedding = MockEmbedding::new(vec![1.0], false);
        let embedding = Arc::new(embedding);
        let indexer = StructuredDocIndexer::new(embedding.clone());
        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: id.to_owned(),
                title: "title".to_owned(),
                author_email: Some("author_email".to_owned()),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let updated_at = chrono::Utc::now();
        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            let updated = indexer
                .presync(&StructuredDocState {
                    id: doc.id().to_string(),
                    updated_at,
                    deleted: false,
                })
                .await
                && indexer.sync(doc).await;
            println!("{updated}");
            updated
        });
        assert!(res);
        indexer.commit();

        let validator = Indexer::new(corpus::STRUCTURED_DOC);

        assert!(validator.is_indexed(id));
        assert!(!validator.has_failed_chunks(id));

        tabby_common::path::set_tabby_root(root);
    }

    #[test]
    #[file_serial(set_tabby_root)]
    fn test_structured_doc_has_attribute_field() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let id = "structured_doc_has_attribute_field";
        let embedding = MockEmbedding::new(vec![1.0], false);
        let embedding = Arc::new(embedding);
        let indexer = StructuredDocIndexer::new(embedding.clone());
        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: id.to_owned(),
                title: "title".to_owned(),
                author_email: Some("author_email".to_owned()),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let res = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { indexer.sync(doc).await });
        assert!(res);
        indexer.commit();

        let validator = Indexer::new(corpus::STRUCTURED_DOC);

        assert!(validator.is_indexed(id));
        assert!(validator.has_attribute_field(id, StructuredDocIndexFields::issue::AUTHOR_EMAIL));

        tabby_common::path::set_tabby_root(root);
    }
}

mod builder_tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use serial_test::file_serial;
    use tabby_common::index::{corpus, IndexSchema};
    use tantivy::schema::Value;
    use temp_testdir::TempDir;

    use super::mock_embedding::MockEmbedding;
    use crate::{
        code::{create_code_builder, intelligence::CodeIntelligence},
        indexer::{TantivyDocBuilder, ToIndexId},
        structured_doc::{
            public::{StructuredDoc, StructuredDocFields, StructuredDocIssueFields},
            StructuredDocBuilder,
        },
        testutils::{get_repository_config, get_rust_source_file, get_tabby_root},
    };

    #[test]
    #[file_serial(set_tabby_root)]
    fn test_builder_code_empty_embedding() {
        let origin_root = tabby_common::path::tabby_root();
        tabby_common::path::set_tabby_root(get_tabby_root());

        let embedding = MockEmbedding::new(vec![], true);
        let builder = Arc::new(create_code_builder(Some(Arc::new(embedding))));

        let repo = get_repository_config();
        let code = CodeIntelligence::compute_source_file(&repo, "commit", &get_rust_source_file())
            .unwrap();
        let index_id = code.to_index_id();

        let (id, s) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { builder.build(code).await });
        assert_eq!(id, index_id.id);

        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            s.buffer_unordered(std::cmp::max(
                std::thread::available_parallelism().unwrap().get() * 2,
                32,
            ))
            .map(|handler| handler.unwrap())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
        });

        // the chunks should be failed as no embedding is provided
        // the last element is the document itself
        assert_eq!(res.len(), 1);
        let doc = res.last().unwrap();

        let schema = IndexSchema::instance();
        let failed_count = doc
            .get_first(schema.field_failed_chunks_count)
            .and_then(|v| v.as_u64())
            .unwrap();

        // the first three are the chunks and failed, counted as 3
        assert_eq!(failed_count, 3);

        tabby_common::path::set_tabby_root(origin_root);
    }

    /// Test that the indexer return the document and none itself
    /// when the embedding is empty
    #[test]
    #[file_serial(set_tabby_root)]
    fn test_builder_empty_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let test_id = "builder_empty_embedding";
        let embedding = MockEmbedding::new(vec![], true);
        let builder = StructuredDocBuilder::new(Arc::new(embedding));
        let tantivy_builder = TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder);

        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: test_id.to_owned(),
                title: "title".to_owned(),
                author_email: Some("author_email".to_owned()),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let (id, s) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { tantivy_builder.build(doc).await });
        assert_eq!(id, test_id);

        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            s.buffer_unordered(std::cmp::max(
                std::thread::available_parallelism().unwrap().get() * 2,
                32,
            ))
            .map(|handler| handler.unwrap())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
        });

        // The last element is the document itself,
        // while the preceding elements are the chunks.
        // Given that the embedding is empty,
        // all chunks should be considered failed and skipped.
        assert_eq!(res.len(), 1);
        let doc = res.last().unwrap();

        let schema = IndexSchema::instance();
        let failed_count = doc
            .get_first(schema.field_failed_chunks_count)
            .and_then(|v| v.as_u64())
            .unwrap();

        assert_eq!(failed_count, 1);

        tabby_common::path::set_tabby_root(root);
    }

    /// Test that the indexer returns the document and the chunk
    /// when the embedding is not empty.
    /// when there are embeddings, the failed count should not be existed
    #[test]
    #[file_serial(set_tabby_root)]
    fn test_builder_with_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let test_id = "builder_with_embedding";
        let embedding = MockEmbedding::new(vec![1.0], false);
        let builder = StructuredDocBuilder::new(Arc::new(embedding));
        let tantivy_builder = TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder);

        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: test_id.to_owned(),
                title: "title".to_owned(),
                author_email: Some("author_email".to_owned()),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let (id, s) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { tantivy_builder.build(doc).await });

        assert_eq!(id, test_id);

        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            s.buffer_unordered(std::cmp::max(
                std::thread::available_parallelism().unwrap().get() * 2,
                32,
            ))
            .collect::<Vec<_>>()
            .await
        });

        // the last element is the document itself
        assert_eq!(res.len(), 2);
        let doc = res[1].as_ref().unwrap().as_ref().unwrap();

        let schema = IndexSchema::instance();
        assert!(doc.get_first(schema.field_failed_chunks_count).is_none());

        tabby_common::path::set_tabby_root(root);
    }
}
