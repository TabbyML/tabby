mod mock_embedding {
    use anyhow::Result;
    use async_trait::async_trait;
    use tabby_inference::Embedding;

    pub struct MockEmbedding {
        result: Vec<f32>,
    }

    impl MockEmbedding {
        pub fn new(result: Vec<f32>) -> Self {
            Self { result }
        }
    }

    #[async_trait]
    impl Embedding for MockEmbedding {
        async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
            if prompt.starts_with("error") {
                Err(anyhow::anyhow!(prompt.to_owned()))
            } else {
                Ok(self.result.clone())
            }
        }
    }
}

mod structured_doc_tests {
    use std::sync::Arc;

    use serial_test::serial;
    use tabby_common::index::corpus;
    use temp_testdir::TempDir;

    use super::mock_embedding::MockEmbedding;
    use crate::{
        indexer::Indexer,
        structured_doc::public::{
            StructuredDoc, StructuredDocFields, StructuredDocIndexer, StructuredDocIssueFields,
        },
    };

    /// the document should be indexed even no embedding is provided
    /// the document itself could be used for search
    #[test]
    #[serial(set_tabby_root)]
    fn test_structured_doc_empty_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let id = "structured_doc_empty_embedding";
        let embedding = MockEmbedding::new(vec![]);
        let embedding = Arc::new(embedding);
        let indexer = StructuredDocIndexer::new(embedding.clone());
        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: id.to_owned(),
                title: "title".to_owned(),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let updated_at = chrono::Utc::now();
        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            let added = indexer.add(updated_at, doc).await;
            println!("{}", added);
            added
        });
        assert!(res);
        indexer.commit();

        let validator = Indexer::new(corpus::STRUCTURED_DOC);
        // Wait for up to 60s for the document to be indexed.
        for _ in 0..10 {
            if validator.is_indexed(id) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        assert!(validator.is_indexed(id));
        assert!(validator.has_failed_chunks(id));

        tabby_common::path::set_tabby_root(root);
    }

    #[test]
    #[serial(set_tabby_root)]
    fn test_structured_doc_with_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let id = "structured_doc_with_embedding";
        let embedding = MockEmbedding::new(vec![1.0]);
        let embedding = Arc::new(embedding);
        let indexer = StructuredDocIndexer::new(embedding.clone());
        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: id.to_owned(),
                title: "title".to_owned(),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let updated_at = chrono::Utc::now();
        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            let added = indexer.add(updated_at, doc).await;
            println!("{}", added);
            added
        });
        assert!(res);
        indexer.commit();

        let validator = Indexer::new(corpus::STRUCTURED_DOC);
        // Wait for up to 60s for the document to be indexed.
        for _ in 0..10 {
            if validator.is_indexed(id) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        assert!(validator.is_indexed(id));
        assert!(!validator.has_failed_chunks(id));

        tabby_common::path::set_tabby_root(root);
    }
}

mod builder_tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use serial_test::serial;
    use tabby_common::index::{corpus, IndexSchema};
    use tantivy::schema::Value;
    use temp_testdir::TempDir;

    use super::mock_embedding::MockEmbedding;
    use crate::{
        indexer::TantivyDocBuilder,
        structured_doc::{
            public::{StructuredDoc, StructuredDocFields, StructuredDocIssueFields},
            StructuredDocBuilder,
        },
    };

    /// Test that the indexer return the document and none itself
    /// when the embedding is empty
    #[test]
    #[serial(set_tabby_root)]
    fn test_builder_empty_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let test_id = "builder_empty_embedding";
        let embedding = MockEmbedding::new(vec![]);
        let builder = StructuredDocBuilder::new(Arc::new(embedding));
        let tantivy_builder = TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder);

        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: test_id.to_owned(),
                title: "title".to_owned(),
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
        // the rest are the chunks
        assert_eq!(res.len(), 2);
        let doc = res[1].as_ref().unwrap().as_ref().unwrap();

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
    #[serial(set_tabby_root)]
    fn test_builder_with_embedding() {
        let root = tabby_common::path::tabby_root();
        let temp_dir = TempDir::default();
        tabby_common::path::set_tabby_root(temp_dir.to_owned());

        let test_id = "builder_with_embedding";
        let embedding = MockEmbedding::new(vec![1.0]);
        let builder = StructuredDocBuilder::new(Arc::new(embedding));
        let tantivy_builder = TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder);

        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: test_id.to_owned(),
                title: "title".to_owned(),
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

mod intelligence_tests {
    use std::path::PathBuf;

    use serial_test::serial;
    use tabby_common::{
        config::{config_index_to_id, CodeRepository},
        path::set_tabby_root,
    };
    use tracing_test::traced_test;

    use crate::code::intelligence::CodeIntelligence;

    fn get_tabby_root() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("testdata");
        path
    }

    fn get_repository_config() -> CodeRepository {
        CodeRepository::new("https://github.com/TabbyML/tabby", &config_index_to_id(0))
    }

    fn get_rust_source_file() -> PathBuf {
        let mut path = get_tabby_root();
        path.push("repositories");
        path.push("https_github.com_TabbyML_tabby");
        path.push("rust.rs");
        path
    }

    #[test]
    #[traced_test]
    #[serial(set_tabby_root)]
    fn test_create_source_file() {
        set_tabby_root(get_tabby_root());
        let config = get_repository_config();
        let source_file = CodeIntelligence::compute_source_file(&config, &get_rust_source_file())
            .expect("Failed to create source file");

        // check source_file properties
        assert_eq!(source_file.language, "rust");
        assert_eq!(source_file.tags.len(), 3);
        assert_eq!(source_file.filepath, "rust.rs");
    }
}
