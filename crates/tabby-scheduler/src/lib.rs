//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
pub mod crawl;
mod indexer;

use async_stream::stream;
pub use code::CodeIndexer;
use crawl::crawl_pipeline;
use doc::SourceDocument;
use futures::StreamExt;
use indexer::{IndexAttributeBuilder, Indexer};

mod doc;
use std::{env, sync::Arc};

use doc::create_web_index;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, info, warn};

pub async fn scheduler<T: RepositoryAccess + 'static>(
    now: bool,
    config: &tabby_common::config::Config,
    access: T,
) {
    if now {
        let repositories = access
            .list_repositories()
            .await
            .expect("Must be able to retrieve repositories for sync");
        scheduler_pipeline(&repositories).await;
        if env::var("TABBY_SCHEDULER_EXPERIMENTAL_DOC_INDEX").is_ok() {
            doc_index_pipeline(config).await;
        }
    } else {
        let access = Arc::new(access);
        let scheduler = JobScheduler::new()
            .await
            .expect("Failed to create scheduler");
        let scheduler_mutex = Arc::new(tokio::sync::Mutex::new(()));
        let _config = config.clone();

        // Every 10 minutes
        scheduler
            .add(
                Job::new_async("0 1/10 * * * *", move |_, _| {
                    let access = access.clone();
                    let scheduler_mutex = scheduler_mutex.clone();
                    Box::pin(async move {
                        let Ok(_guard) = scheduler_mutex.try_lock() else {
                            warn!("Scheduler job overlapped, skipping...");
                            return;
                        };

                        let repositories = access
                            .list_repositories()
                            .await
                            .expect("Must be able to retrieve repositories for sync");

                        scheduler_pipeline(&repositories).await;
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

async fn scheduler_pipeline(repositories: &[RepositoryConfig]) {
    let mut code = CodeIndexer::default();
    for repository in repositories {
        code.refresh(repository).await;
    }

    code.garbage_collection(repositories);
}

async fn doc_index_pipeline(config: &tabby_common::config::Config) {
    let Some(index_config) = &config.experimental.doc else {
        return;
    };

    let Some(embedding_config) = &config.model.embedding else {
        return;
    };

    debug!("Starting doc index pipeline...");
    let embedding = llama_cpp_server::create_embedding(embedding_config).await;
    for url in &index_config.start_urls {
        let embedding = embedding.clone();
        stream! {
            let mut num_docs = 0;
            let doc_index = create_web_index(embedding.clone());
            for await doc in crawl_pipeline(url).await {
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
        .collect::<Vec<_>>()
        .await;
    }
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
