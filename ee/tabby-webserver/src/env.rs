pub fn demo_mode() -> bool {
    matches!(
        &*std::env::var("TABBY_WEBSERVER_DEMO_MODE").unwrap_or_default(),
        "true" | "1"
    )
}
