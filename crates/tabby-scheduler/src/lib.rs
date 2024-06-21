//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
pub mod crawl;
mod indexer;

pub use code::CodeIndexer;
use crawl::crawl_pipeline;
use doc::create_web_index;
pub use doc::{DocIndexer, WebDocument};
use futures::{Future, StreamExt};
use indexer::{IndexAttributeBuilder, Indexer};
use tabby_inference::Embedding;

mod doc;
use std::sync::Arc;

use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, info, warn};

use crate::doc::SourceDocument;

pub async fn scheduler(now: bool, config: &tabby_common::config::Config) {
    if now {
        scheduler_pipeline(config).await;
    } else {
        let scheduler = JobScheduler::new()
            .await
            .expect("Failed to create scheduler");
        let scheduler_mutex = Arc::new(tokio::sync::Mutex::new(()));
        let config = config.clone();

        // Every 10 minutes
        scheduler
            .add(
                Job::new_async("0 1/10 * * * *", move |_, _| {
                    let config = config.clone();
                    let scheduler_mutex = scheduler_mutex.clone();
                    Box::pin(async move {
                        let Ok(_guard) = scheduler_mutex.try_lock() else {
                            warn!("Scheduler job overlapped, skipping...");
                            return;
                        };

                        scheduler_pipeline(&config).await;
                    })
                })
                .expect("Failed to create job"),
            )
            .await
            .expect("Failed to add job");

        info!("Scheduler activated...");
        scheduler.start().await.expect("Failed to start scheduler");

        // Sleep 10 years (indefinitely)
        tokio::time::sleep(tokio::time::Duration::from_secs(3600 * 24 * 365 * 10)).await;
    }
}

async fn scheduler_pipeline(config: &tabby_common::config::Config) {
    let embedding_config = &config.model.embedding;

    let embedding = llama_cpp_server::create_embedding(embedding_config).await;

    let repositories = &config.repositories;

    let mut code = CodeIndexer::default();
    for repository in repositories {
        code.refresh(embedding.clone(), repository).await;
    }

    code.garbage_collection(repositories);
}

pub async fn crawl_index_docs<F>(
    urls: &[String],
    embedding: Arc<dyn Embedding>,
    on_process_url: impl Fn(String) -> F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()>,
{
    for url in urls {
        debug!("Starting doc index pipeline for {url}");
        let embedding = embedding.clone();
        let mut num_docs = 0;
        let doc_index = create_web_index(embedding.clone());

        let mut pipeline = Box::pin(crawl_pipeline(url).await?);
        while let Some(doc) = pipeline.next().await {
            on_process_url(doc.url.clone()).await;
            let source_doc = SourceDocument {
                id: doc.url.clone(),
                title: doc.metadata.title.unwrap_or_default(),
                link: doc.url,
                body: doc.markdown,
            };

            num_docs += 1;
            doc_index.add(source_doc).await;
        }
        info!("Crawled {} documents from '{}'", num_docs, url);
        doc_index.commit();
    }
    Ok(())
}

mod tantivy_utils {
    use std::{fs, path::Path};

    use tantivy::{directory::MmapDirectory, schema::Schema, Index};
    use tracing::{debug, warn};

    pub fn open_or_create_index(code: &Schema, path: &Path) -> (bool, Index) {
        let (recreated, index) = match open_or_create_index_impl(code, path) {
            Ok(index) => (false, index),
            Err(err) => {
                warn!(
                    "Failed to open index repositories: {}, removing index directory '{}'...",
                    err,
                    path.display()
                );
                fs::remove_dir_all(path).expect("Failed to remove index directory");

                debug!("Reopening index repositories...");
                (
                    true,
                    open_or_create_index_impl(code, path).expect("Failed to open index"),
                )
            }
        };
        (recreated, index)
    }

    fn open_or_create_index_impl(code: &Schema, path: &Path) -> tantivy::Result<Index> {
        fs::create_dir_all(path).expect("Failed to create index directory");
        let directory = MmapDirectory::open(path).expect("Failed to open index directory");
        Index::open_or_create(directory, code.clone())
    }
}
