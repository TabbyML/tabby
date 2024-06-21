#[tokio::main]
async fn main() {
    // PORT=4443 npx https-localhost
    // certificate will automatically load into system (through mkcert)
    let resp = reqwest::get("https://localhost:4433").await.unwrap();

    // Should output 404
    println!("Status: {}", resp.status());
}
