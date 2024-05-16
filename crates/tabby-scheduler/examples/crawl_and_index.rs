use async_stream::stream;
use futures::StreamExt;
use llama_cpp_server::create_embedding;
use tabby_common::registry::{parse_model_id, ModelRegistry};
use tabby_scheduler::{crawl::crawl_pipeline, DocIndex, SourceDocument};
use tracing::debug;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("tabby=debug,crawl_and_index=debug")
        .init();

    let (registry, name) = parse_model_id("wsxiaoys/Nomic-Embed-Text");
    let registry = ModelRegistry::new(registry).await;
    let model_path = registry.get_model_path(name).display().to_string();

    let mut doc_index = DocIndex::new(create_embedding(false, &model_path, 1).await);
    let mut cnt = 0;
    stream! {
        for await doc in crawl_pipeline("https://tabby.tabbyml.com/").await {
            debug!("Title: {:?}", doc.metadata.title);
            debug!("Description: {:?}", doc.metadata.description);
            debug!("URL: {}\n", doc.url);
            cnt += 1;
            if cnt >= 5 {
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

        debug!("Committing index...");
        doc_index.commit();
    }
    .collect::<()>()
    .await;
}
