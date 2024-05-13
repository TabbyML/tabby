use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use tabby_inference::Embedding;
use tabby_scheduler::{crawl::crawl_pipeline, DocIndex, SourceDocument};
use tracing::debug;

#[tokio::main]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("tabby=debug,crawl_and_index=debug")
        .init();

    let mut doc_index = DocIndex::new(Arc::new(FakeEmbedding));
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

struct FakeEmbedding;

#[async_trait]
impl Embedding for FakeEmbedding {
    async fn embed(&self, _: &str) -> anyhow::Result<Vec<f32>> {
        let mut embedding = vec![0.0; 512];
        embedding[3] = 1.0;
        embedding[128] = 1.0;
        Ok(embedding)
    }
}
