pub fn get_download_host() -> String {
    std::env::var("TABBY_DOWNLOAD_HOST").unwrap_or_else(|_| "huggingface.co".to_string())
}

pub fn get_huggingface_mirror_host() -> Option<String> {
    std::env::var("TABBY_HUGGINGFACE_HOST_OVERRIDE").ok()
}

// for debug only
pub fn use_local_model_json() -> bool {
    std::env::var("TABBY_USE_LOCAL_MODEL_JSON").is_ok()
}