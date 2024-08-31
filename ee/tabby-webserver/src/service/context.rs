use std::sync::Arc;

use tabby_schema::{
    context::{ContextInfo, ContextService},
    repository::RepositoryService,
    web_documents::WebDocumentService,
    Result,
};

use super::answer::AnswerService;

struct ContextServiceImpl {
    repository: Arc<dyn RepositoryService>,
    web_document: Arc<dyn WebDocumentService>,
    answer: Option<Arc<AnswerService>>,
}

#[async_trait::async_trait]
impl ContextService for ContextServiceImpl {
    async fn read(&self) -> Result<ContextInfo> {
        let mut sources: Vec<_> = self
            .repository
            .repository_list()
            .await?
            .into_iter()
            .map(Into::into)
            .collect();

        sources.extend(
            self.web_crawler
                .list_web_crawler_urls(None, None, None, None)
                .await?
                .into_iter()
                .map(Into::into),
        );

        sources.extend(
            self.web_document
                .list_custom_web_documents(None, None, None, None, None)
                .await?
                .into_iter()
                .map(Into::into),
        );

        sources.extend(
            self.web_document
                .list_preset_web_documents(None, None, None, None, None, Some(true))
                .await?
                .into_iter()
                .map(Into::into),
        );

        let info = ContextInfo {
            sources,
            can_search_public: self
                .answer
                .as_ref()
                .map(|x| x.can_search_public())
                .unwrap_or_default(),
        };

        Ok(info)
    }
}

pub fn create(
    repository: Arc<dyn RepositoryService>,
    web_crawler: Arc<dyn WebCrawlerService>,
    web_document: Arc<dyn WebDocumentService>,
    answer: Option<Arc<AnswerService>>,
) -> impl ContextService {
    ContextServiceImpl {
        repository,
        web_crawler,
        web_document,
        answer,
    }
}
