use async_trait::async_trait;
use juniper::{GraphQLObject, ID};

use super::{user_group::UserGroup, Context, Result};

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct SourceIdAccessPolicy {
    pub source_id: String,
    pub read: Vec<UserGroup>,
}

#[async_trait]
pub trait AccessPolicyService: Sync + Send {
    async fn list_source_id_read_access(&self, source_id: &str) -> Result<Vec<UserGroup>>;
    async fn grant_source_id_read_access(&self, source_id: &str, user_group_id: &ID) -> Result<()>;
    async fn revoke_source_id_read_access(&self, source_id: &str, user_group_id: &ID)
        -> Result<()>;
}
