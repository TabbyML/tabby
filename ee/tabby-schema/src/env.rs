pub fn is_demo_mode() -> bool {
    std::env::var("TABBY_WEBSERVER_DEMO_MODE").is_ok()
}

pub fn gitlab_ssl_cert() -> Option<String> {
    std::env::var("GITLAB_SSL_CERT").ok()
}

pub fn gitlab_ssl_insecure() -> bool {
    std::env::var("GITLAB_SSL_INSECURE").is_ok()
}
