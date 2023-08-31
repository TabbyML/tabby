use std::collections::HashMap;

use lazy_static::lazy_static;

pub fn reduce_language_if_needed(language: &str) -> &str {
    return match LANGUAGE_REDUCE_MAP.get(language) {
        Some(res) => res,
        None => language,
    };
}

lazy_static! {
    static ref LANGUAGE_REDUCE_MAP: HashMap<&'static str, &'static str> = HashMap::from([
        ("javascript", "js_ts"),
        ("typescript", "js_ts"),
        ("jsx", "js_ts"),
        ("tsx", "js_ts"),
    ]);
}
