use juniper::ID;
use std::marker::PhantomData;
use strum::IntoEnumIterator;
use tabby_schema::{
    integration::IntegrationKind, repository::ProvidedRepository, AsRowid, DbEnum, Result,
};
use url::Url;

use async_trait::async_trait;
use tabby_db::DbConn;
use tabby_schema::repository::{RepositoryProvider, ThirdPartyRepositoryService};

use crate::service::graphql_pagination_to_filter;

struct ThirdPartyRepositoryServiceImpl {
    db: DbConn,
}

#[async_trait]
impl ThirdPartyRepositoryService for ThirdPartyRepositoryServiceImpl {
    async fn list_repositories(
        &self,
        integration_ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<ProvidedRepository>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let integration_ids = integration_ids
            .into_iter()
            .flatten()
            .map(|id| id.as_rowid())
            .collect::<Result<Vec<_>, _>>()?;

        let kind = kind.map(|kind| kind.as_enum_str().to_string());

        Ok(self
            .db
            .list_provided_repositories(integration_ids, kind, active, limit, skip_id, backwards)
            .await?
            .into_iter()
            .map(ProvidedRepository::try_from)
            .collect::<Result<_, _>>()?)
    }

    async fn update_repository_active(&self, id: ID, active: bool) -> Result<()> {
        self.db
            .update_provided_repository_active(id.as_rowid()?, active)
            .await?;
        Ok(())
    }

    async fn list_active_git_urls(&self) -> Result<Vec<String>> {
        let mut urls = vec![];
    }
}

fn format_authenticated_url(
    kind: &IntegrationKind,
    git_url: &str,
    access_token: &str,
) -> Result<String> {
    let mut url = Url::parse(git_url).map_err(anyhow::Error::from)?;
    match kind {
        IntegrationKind::Github => {
            let _ = url.set_username(access_token);
        }
        IntegrationKind::Gitlab => {
            let _ = url.set_username("oauth2");
            let _ = url.set_password(Some(access_token));
        }
    }
    Ok(url.to_string())
}
