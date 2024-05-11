mod chat;
mod completion;
mod embedding;

pub use chat::create as create_chat;
pub use completion::create;
pub use embedding::create as create_embedding;
use serde_json::Value;

pub(crate) fn get_param(params: &Value, key: &str) -> String {
    params
        .get(key)
        .unwrap_or_else(|| panic!("Missing {} field", key))
        .as_str()
        .expect("Type unmatched")
        .to_owned()
}

pub(crate) fn get_optional_param(params: &Value, key: &str) -> Option<String> {
    params
        .get(key)
        .map(|x| x.as_str().expect("Type unmatched").to_owned())
}
