mod api;
mod imp;

pub use api::*;

pub fn create_local() -> BoxCodeSearch {
    Box::new(imp::CodeSearchService::new())
}
