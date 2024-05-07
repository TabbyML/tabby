use async_stream::stream;
use futures::StreamExt;
use tabby_scheduler::crawl::crawl_pipeline;

#[tokio::main]
async fn main() {
    let mut cnt = 3;
    stream! {
        println!("Crawling https://tabby.tabbyml.com/");
        for await doc in crawl_pipeline("https://tabby.tabbyml.com/").await {
            println!("Title: {:?}", doc.metadata.title);
            println!("Description: {:?}", doc.metadata.description);
            println!("URL: {}\n", doc.url);
            println!("Markdown: {}", doc.markdown);
            cnt = cnt - 1;
            if cnt <= 0 {
                break;
            }
        }
    }
    .collect::<()>()
    .await;
}
