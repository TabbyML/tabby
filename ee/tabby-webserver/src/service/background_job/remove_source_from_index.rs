


use tabby_scheduler::{remove_source_from_index};

use super::helper::Job;

pub struct RemoveSourceFromIndex {
    corpus: String,
    source_id: String,
}

impl Job for RemoveSourceFromIndex {
    const NAME: &'static str = "remove_source_from_index";
}

impl RemoveSourceFromIndex {
    pub fn new(corpus: String, source_id: String) -> Self {
        Self { corpus, source_id }
    }

    pub async fn run(self) -> tabby_schema::Result<()> {
        remove_source_from_index(&self.corpus, &self.source_id);
        Ok(())
    }
}
