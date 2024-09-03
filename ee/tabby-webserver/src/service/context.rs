use std::sync::Arc;

use juniper::ID;
use tabby_schema::{
    context::{ContextInfo, ContextKind, ContextService, ContextSource},
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
        let mut sources: Vec<_> = self
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
            let source_id = "web";
            sources.push(ContextSource {
                id: ID::from(source_id.to_owned()),
                kind: ContextKind::Web,
                source_id: source_id.into(),
                display_name: "Web".to_string(),
            });
        }

        if let Some(policy) = policy {
            // Keep only sources that the user has access to.
            sources.retain(|x| policy.check_read_source(&x.source_id).is_ok());
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
