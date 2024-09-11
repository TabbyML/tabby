use std::sync::Arc;

use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    access_policy::AccessPolicyService, bail, context::ContextService, user_group::UserGroup,
    AsRowid, Result,
};

use super::UserGroupExt;

struct AccessPolicyServiceImpl {
    db: DbConn,
    context: Arc<dyn ContextService>,
}

#[async_trait::async_trait]
impl AccessPolicyService for AccessPolicyServiceImpl {
    async fn list_source_id_read_access(&self, source_id: &str) -> Result<Vec<UserGroup>> {
        let mut user_groups = Vec::new();
        for x in self
            .db
            .list_source_id_read_access_user_groups(source_id)
            .await?
        {
            user_groups.push(UserGroup::new(self.db.clone(), x).await?)
        }

        Ok(user_groups)
    }

    async fn grant_source_id_read_access(&self, source_id: &str, user_group_id: &ID) -> Result<()> {
        let context_info = self.context.read(None).await?;
        let helper = context_info.helper();
        if !helper.can_access_source_id(source_id) {
            bail!("source_id {} not found", source_id)
        }

        self.db
            .upsert_source_id_read_access_policy(source_id, user_group_id.as_rowid()?)
            .await?;
        Ok(())
    }

    async fn revoke_source_id_read_access(
        &self,
        source_id: &str,
        user_group_id: &ID,
    ) -> Result<()> {
        self.db
            .delete_source_id_read_access_policy(source_id, user_group_id.as_rowid()?)
            .await?;
        Ok(())
    }
}

pub fn create(db: DbConn, context: Arc<dyn ContextService>) -> impl AccessPolicyService {
    AccessPolicyServiceImpl { db, context }
}
