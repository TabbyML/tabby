use std::{env, sync::Arc};

use async_stream::stream;
use futures::StreamExt;
use llama_cpp_server::LlamaCppServer;
use tabby_scheduler::{crawl::crawl_pipeline, DocIndex, SourceDocument};
use tracing::debug;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("tabby=debug,crawl_and_index=debug")
        .init();

    let model_path = env::var("MODEL_PATH").expect("MODEL_PATH is not set");

    let mut doc_index = DocIndex::new(Arc::new(LlamaCppServer::new(false, true, &model_path, 1)));
    let mut cnt = 0;
    stream! {
        for await doc in crawl_pipeline("https://tabby.tabbyml.com/").await {
            debug!("Title: {:?}", doc.metadata.title);
            debug!("Description: {:?}", doc.metadata.description);
            debug!("URL: {}\n", doc.url);
            cnt += 1;
            if cnt >= 3 {
                break;
            }

            let id = cnt.to_string();
            debug!("Adding document {} to index...", id);
            let source_doc = SourceDocument {
                id,
                title: doc.metadata.title.unwrap_or_default(),
                link: doc.url,
                body: doc.markdown,
            };

            doc_index.add(source_doc).await;
        }

        doc_index.commit();
    }
    .collect::<()>()
    .await;
}
