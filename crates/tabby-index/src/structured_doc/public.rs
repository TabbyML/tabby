use std::{future::Future, sync::Arc};

use anyhow::Result;
use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use tabby_common::index::{corpus, structured_doc::fields as StructuredDocIndexFields};
use tabby_inference::Embedding;
use tracing::debug;

pub use super::types::{
    commit::CommitDocument as StructuredDocCommitFields,
    ingested::IngestedDocument as StructuredDocIngestedFields,
    issue::IssueDocument as StructuredDocIssueFields,
    page::PageDocument as StructuredDocPageFields,
    pull::PullDocument as StructuredDocPullDocumentFields,
    web::WebDocument as StructuredDocWebFields, StructuredDoc, StructuredDocFields, KIND_COMMIT,
    KIND_INGESTED,
};
use super::{create_structured_doc_builder, types::BuildStructuredDoc};
use crate::{indexer::TantivyDocBuilder, Indexer};

/// StructuredDocState tracks the state of the document source.
/// It helps determine whether the document should be updated or deleted.
pub struct StructuredDocState {
    // id is the unique identifier of the document.
    // It is used to track the document in the indexer.
    pub id: String,

    // updated_at is the time when the document was last updated.
    // when the updated_at is earlier than the document's index time,
    // the update will be skipped.
    pub updated_at: DateTime<Utc>,

    // deleted indicates whether the document should be removed from the indexer.
    // For instance, a closed pull request will be marked as deleted,
    // prompting the indexer to remove it from the index.
    pub deleted: bool,
}

pub struct StructuredDocIndexer {
    builder: TantivyDocBuilder<StructuredDoc>,
    indexer: Indexer,
}

impl StructuredDocIndexer {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let builder = create_structured_doc_builder(embedding);
        let indexer = Indexer::new(corpus::STRUCTURED_DOC);
        Self { indexer, builder }
    }

    // Runs pre-sync checks to determine if the document needs to be updated.
    // Returns false if `sync` is not required to be called.
    pub async fn presync(&self, state: &StructuredDocState) -> bool {
        if state.deleted {
            self.indexer.delete(&state.id);
            return false;
        }

        if self.indexer.is_indexed_after(&state.id, state.updated_at)
            && !self.indexer.has_failed_chunks(&state.id)
        {
            return false;
        };

        true
    }

    // The sync process updates the document in the indexer incrementally.
    // It first determines whether the document requires an update.
    //
    // If an update is needed, it checks the deletion state of the document.
    // If the document is marked as deleted, it will be removed.
    // Next, the document is rebuilt, the original is deleted, and the newly indexed document is added.
    pub async fn sync(&self, document: StructuredDoc) -> bool {
        if !self.require_updates(&document).await {
            return false;
        }

        stream! {
            let (id, s) = self.builder.build(document).await;
            self.indexer.delete(&id);

            for await doc in s.buffer_unordered(std::cmp::max(std::thread::available_parallelism().unwrap().get() * 2, 32)) {
                if let Ok(Some(doc)) = doc {
                    self.indexer.add(doc).await;
                }
            }
        }.count().await;
        true
    }

    pub async fn delete(&self, id: &str) -> bool {
        if self.indexer.is_indexed(id) {
            self.indexer.delete(id);
            true
        } else {
            false
        }
    }

    pub async fn count_doc(&self, source_id: &str, kind: &str) -> Result<usize> {
        let attributes = vec![(StructuredDocIndexFields::KIND, kind)];
        self.indexer
            .count_doc_by_attribute(source_id, &attributes)
            .await
    }

    pub async fn list_latest_ids(
        &self,
        source_id: &str,
        kind: &str,
        datetime_field: &str,
        offset: usize,
    ) -> Result<Vec<String>> {
        self.indexer
            .list_latest_ids(
                source_id,
                &vec![(StructuredDocIndexFields::KIND, kind)],
                datetime_field,
                offset,
            )
            .await
    }

    pub fn commit(self) {
        self.indexer.commit();
    }

    async fn require_updates(&self, document: &StructuredDoc) -> bool {
        if document.should_skip() {
            return false;
        }

        if self.should_backfill(document) {
            return true;
        }

        if let StructuredDocFields::Commit(_commit) = &document.fields {
            if self.indexer.get_doc(document.id()).await.is_ok() {
                return false;
            }
        }

        true
    }

    fn should_backfill(&self, document: &StructuredDoc) -> bool {
        // v0.22.0 add the author field to the issue and pull documents.
        match &document.fields {
            StructuredDocFields::Issue(issue) => {
                if issue.author_email.is_some()
                    && !self.indexer.has_attribute_field(
                        document.id(),
                        StructuredDocIndexFields::issue::AUTHOR_EMAIL,
                    )
                {
                    return true;
                }
            }
            StructuredDocFields::Pull(pull) => {
                if pull.author_email.is_some()
                    && !self.indexer.has_attribute_field(
                        document.id(),
                        StructuredDocIndexFields::pull::AUTHOR_EMAIL,
                    )
                {
                    return true;
                }
            }
            _ => (),
        }

        false
    }
}

pub struct StructuredDocGarbageCollector {
    indexer: Indexer,
}

impl Default for StructuredDocGarbageCollector {
    fn default() -> Self {
        Self {
            indexer: Indexer::new(corpus::STRUCTURED_DOC),
        }
    }
}

impl StructuredDocGarbageCollector {
    pub async fn run<F, Fut>(self, should_keep_ingested: F) -> anyhow::Result<()>
    where
        F: Fn(String, String) -> Fut + Send + Sync,
        Fut: Future<Output = bool> + Send,
    {
        stream! {
            let mut num_to_delete = 0;

            for await (source_id, id) in self.indexer.iter_ids() {
                let kind = if let Ok(Some(kind)) = self.indexer.get_doc_kind(&id).await {
                    kind
                } else {
                    continue
                };

                if kind.as_str() == KIND_INGESTED {
                    let doc_id = if let Some(doc_id) = id.strip_prefix(&format!("{source_id}/")) {
                        doc_id
                    } else {
                        debug!("ingested doc has incorrect id format, deleting, id: {}, source: {}", id, source_id);
                        num_to_delete += 1;
                        self.indexer.delete(&id);
                        continue;
                    };
                    if !should_keep_ingested(source_id.clone(), doc_id.to_owned()).await {
                        num_to_delete += 1;
                        self.indexer.delete(&id);
                    }
                }
            }

            self.indexer.commit();
            logkit::info!("Finished garbage collection for structured doc index: {num_to_delete} items removed");
        }.collect::<()>().await;

        Ok(())
    }
}
