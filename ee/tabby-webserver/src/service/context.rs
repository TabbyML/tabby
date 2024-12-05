use std::sync::Arc;

use tabby_schema::{
    context::{ContextInfo, ContextService, ContextSourceValue, WebContextSource},
    policy::AccessPolicy,
    repository::RepositoryService,
    web_documents::WebDocumentService,
    Result,
};

struct ContextServiceImpl {
    repository: Arc<dyn RepositoryService>,
    web_document: Arc<dyn WebDocumentService>,
    can_search_public_web: bool,
}

#[async_trait::async_trait]
impl ContextService for ContextServiceImpl {
    async fn read(&self, policy: Option<&AccessPolicy>) -> Result<ContextInfo> {
        let mut sources: Vec<ContextSourceValue> = self
            .repository
            .repository_list(policy)
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

        if self.can_search_public_web {
            sources.push(WebContextSource.into());
        }

        if let Some(policy) = policy {
            // Keep only sources that the user has access to.
            let mut filtered_sources = vec![];
            for source in sources {
                if policy.check_read_source(&source.source_id()).await.is_ok() {
                    filtered_sources.push(source);
                }
            }
            sources = filtered_sources
        }

        Ok(ContextInfo { sources })
    }
}

pub fn create(
    repository: Arc<dyn RepositoryService>,
    web_document: Arc<dyn WebDocumentService>,
    can_search_public_web: bool,
) -> impl ContextService {
    ContextServiceImpl {
        repository,
        web_document,
        can_search_public_web,
    }
}
