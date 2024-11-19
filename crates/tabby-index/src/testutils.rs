mod structured_doc_tests {
    use std::sync::Arc;

    use serial_test::serial;
    use tabby_common::index::corpus;
    use tabby_inference::MockEmbedding;

    use crate::{
        indexer::Indexer,
        structured_doc::public::{
            StructuredDoc, StructuredDocFields, StructuredDocIndexer, StructuredDocIssueFields,
        },
    };

    #[test]
    #[serial(tabby_index)]
    fn test_structured_doc_empty_embedding() {
        let embedding = MockEmbedding::new(vec![]);
        let embedding = Arc::new(embedding);
        let indexer = StructuredDocIndexer::new(embedding.clone());
        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: "empty_embedding".to_owned(),
                title: "title".to_owned(),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let updated_at = chrono::Utc::now();
        let res = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { indexer.add(updated_at, doc).await });
        assert!(!res);
        indexer.commit();

        let validator = Indexer::new(corpus::STRUCTURED_DOC);
        assert!(!validator.is_indexed("empty_embedding"));
    }

    #[test]
    #[serial(tabby_index)]
    fn test_structured_doc_with_embedding() {
        let embedding = MockEmbedding::new(vec![1.0]);
        let embedding = Arc::new(embedding);
        let indexer = StructuredDocIndexer::new(embedding.clone());
        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: "with_embedding".to_owned(),
                title: "title".to_owned(),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let updated_at = chrono::Utc::now();
        let res = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { indexer.add(updated_at, doc).await });
        assert!(res);
        indexer.commit();

        let validator = Indexer::new(corpus::STRUCTURED_DOC);
        assert!(validator.is_indexed("with_embedding"));
    }
}

mod indexer_tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use tabby_common::index::corpus;
    use tabby_inference::MockEmbedding;

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
    fn test_indexer_empty_embedding() {
        let embedding = MockEmbedding::new(vec![]);
        let builder = StructuredDocBuilder::new(Arc::new(embedding));
        let indexer = TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder);

        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: "empty_embedding".to_owned(),
                title: "title".to_owned(),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let (id, s) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { indexer.build(doc).await });
        assert_eq!(id, "empty_embedding");

        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            s.buffer_unordered(std::cmp::max(
                std::thread::available_parallelism().unwrap().get() * 2,
                32,
            ))
            .collect::<Vec<_>>()
            .await
        });

        // the first element is the document itself
        assert_eq!(res.len(), 2);
        // the second element is the chunk,
        // which is empty as the MockEmbedding returns empty
        assert!(res[1].is_ok());
        assert!(res[1].as_ref().unwrap().is_none());
    }

    /// Test that the indexer returns the document and the chunk
    /// when the embedding is not empty
    #[test]
    fn test_indexer_with_embedding() {
        let embedding = MockEmbedding::new(vec![1.0]);
        let builder = StructuredDocBuilder::new(Arc::new(embedding));
        let indexer = TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder);

        let doc = StructuredDoc {
            source_id: "source".to_owned(),
            fields: StructuredDocFields::Issue(StructuredDocIssueFields {
                link: "with_embedding".to_owned(),
                title: "title".to_owned(),
                body: "body".to_owned(),
                closed: false,
            }),
        };

        let (id, s) = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { indexer.build(doc).await });

        assert_eq!(id, "with_embedding");

        let res = tokio::runtime::Runtime::new().unwrap().block_on(async {
            s.buffer_unordered(std::cmp::max(
                std::thread::available_parallelism().unwrap().get() * 2,
                32,
            ))
            .collect::<Vec<_>>()
            .await
        });

        // the first element is the document itself
        assert_eq!(res.len(), 2);
        assert!(res[1].is_ok());
        assert!(res[1].as_ref().unwrap().is_some());
    }
}
