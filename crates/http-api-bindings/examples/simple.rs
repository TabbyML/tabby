use std::env;

use http_api_bindings::vertex_ai::VertexAIEngine;
use tabby_inference::{TextGeneration, TextGenerationOptionsBuilder};
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let api_endpoint = env::var("API_ENDPOINT").expect("API_ENDPOINT not set");
    let authorization = env::var("AUTHORIZATION").expect("AUTHORIZATION not set");
    let engine = VertexAIEngine::create(&api_endpoint, &authorization);

    let options = TextGenerationOptionsBuilder::default()
        .max_decoding_length(32)
        .build()
        .unwrap();
    let prompt = "def fib(n)";
    let text = engine.generate(prompt, options).await;
    info!("{}{}", prompt, text);
}
