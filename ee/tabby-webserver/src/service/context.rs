use std::sync::Arc;

use juniper::ID;
use tabby_schema::{
    context::{ContextInfo, ContextKind, ContextService, ContextSource},
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

        if self.answer.as_ref().map(|x| x.can_search_public_web()).unwrap_or_default() {
            let source_id = "web";
            sources.push(ContextSource {
                id: ID::from(source_id.to_owned()),
                kind: ContextKind::Web,
                source_id: source_id.into(),
                display_name: "Web".to_string(),
            });
        }

        Ok(ContextInfo { sources })
    }
}

pub fn create(
    repository: Arc<dyn RepositoryService>,
    web_document: Arc<dyn WebDocumentService>,
    answer: Option<Arc<AnswerService>>,
) -> impl ContextService {
    ContextServiceImpl {
        repository,
        web_document,
        answer,
    }
}
