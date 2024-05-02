pub fn is_demo_mode() -> bool {
    std::env::var("TABBY_WEBSERVER_DEMO_MODE").is_ok()
}
