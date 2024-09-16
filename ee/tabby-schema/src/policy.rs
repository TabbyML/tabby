use juniper::ID;
use tabby_db::DbConn;

use crate::{user_group::UpsertUserGroupMembershipInput, AsRowid, CoreError, Result};

#[derive(Clone)]
pub struct AccessPolicy {
    db: DbConn,
    user_id: ID,
    is_admin: bool,
}

impl std::fmt::Debug for AccessPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccessPolicy")
            .field("user_id", &self.user_id)
            .field("is_admin", &self.is_admin)
            .finish()
    }
}

impl AccessPolicy {
    pub fn new(db: DbConn, user_id: &ID, is_admin: bool) -> Self {
        Self {
            db,
            user_id: user_id.to_owned(),
            is_admin,
        }
    }

    pub fn check_delete_thread_messages(&self, user_id: &ID) -> Result<()> {
        if self.user_id != *user_id {
            return Err(CoreError::Forbidden(
                "You cannot delete messages in a thread that you do not own.",
            ));
        }

        Ok(())
    }

    pub fn check_update_thread_persistence(&self, user_id: &ID) -> Result<()> {
        if self.user_id != *user_id {
            return Err(CoreError::Forbidden(
                "You cannot update the persistence of a thread that you do not own.",
            ));
        }

        Ok(())
    }

    pub fn check_read_analytic(&self, users: &[ID]) -> Result<()> {
        const ERROR: Result<()> = Err(CoreError::Forbidden(
            "You must be admin to read other users' analytic data",
        ));

        if users.is_empty() && !self.is_admin {
            return ERROR;
        }

        if !self.is_admin {
            for id in users {
                if self.user_id != *id {
                    return ERROR;
                }
            }
        }

        Ok(())
    }

    pub async fn check_read_source(&self, source_id: &str) -> Result<()> {
        let allow = self
            .db
            .allow_read_source(self.user_id.as_rowid()?, source_id)
            .await?;
        if !allow {
            return Err(CoreError::Forbidden(
                "You are not allowed to read this source",
            ));
        }

        Ok(())
    }

    pub async fn check_upsert_user_group_membership(
        &self,
        input: &UpsertUserGroupMembershipInput,
    ) -> Result<()> {
        if self.is_admin {
            return Ok(());
        }

        // User group admin can change membership within their group
        if !self
            .is_user_group_admin(&input.user_group_id, &self.user_id)
            .await
        {
            return Err(CoreError::Forbidden(
                "You are not allowed to update this user group membership",
            ));
        }

        if input.is_group_admin {
            return Err(CoreError::Forbidden(
                "You are not allowed to grant group admin privileges",
            ));
        }

        if self
            .is_user_group_admin(&input.user_group_id, &input.user_id)
            .await
        {
            return Err(CoreError::Forbidden(
                "You are not allowed to modify group admin privileges",
            ));
        }

        Ok(())
    }

    pub async fn check_delete_user_group_membership(
        &self,
        user_group_id: &ID,
        user_id: &ID,
    ) -> Result<()> {
        if self.is_admin {
            return Ok(());
        }

        let err = Err(CoreError::Forbidden(
            "You are not allowed to modify group membership",
        ));

        if !self.is_user_group_admin(user_group_id, &self.user_id).await {
            return err;
        }

        // Cannot remove admin from group
        if self.is_user_group_admin(user_group_id, user_id).await {
            return err;
        }

        Ok(())
    }

    async fn is_user_group_admin(&self, user_group_id: &ID, user_id: &ID) -> bool {
        self.is_user_group_admin_impl(user_group_id, user_id)
            .await
            .unwrap_or_default()
    }

    async fn is_user_group_admin_impl(&self, user_group_id: &ID, user_id: &ID) -> Result<bool> {
        let x = self
            .db
            .list_user_group_memberships(user_group_id.as_rowid()?, Some(user_id.as_rowid()?))
            .await?;
        Ok(x.first().is_some_and(|x| x.is_group_admin))
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::testutils;

    use super::*;
    use crate::AsID;

    #[tokio::test]
    async fn test_check_read_source() {
        let db = DbConn::new_in_memory().await.unwrap();
        let source_id = "source_id";

        let user_id1 = testutils::create_user(&db).await;
        let user_id2 = testutils::create_user2(&db).await;

        let policy1 = AccessPolicy::new(db.clone(), &user_id1.as_id(), false);
        let policy2 = AccessPolicy::new(db.clone(), &user_id2.as_id(), false);

        // 1. Setup user group
        let user_group_id = db.create_user_group("test").await.unwrap();
        db.upsert_user_group_membership(user_id1, user_group_id, false)
            .await
            .unwrap();

        // For source id without any access policies, it's public (readable by all users)
        assert!(policy1.check_read_source(source_id).await.is_ok());
        assert!(policy2.check_read_source(source_id).await.is_ok());

        // 2. add user_group to source id's policy, making it private
        db.upsert_source_id_read_access_policy(source_id, user_group_id)
            .await
            .unwrap();

        // user2 won't be able to access source, while user1 can.
        assert!(policy2.check_read_source(source_id).await.is_err());
        assert!(policy1.check_read_source(source_id).await.is_ok());

        // 3. remove user1 from user_group
        db.delete_user_group_membership(user_id1, user_group_id)
            .await
            .unwrap();

        // user1 won't be able to acces source either now.
        assert!(policy1.check_read_source(source_id).await.is_err());

        // 4. delete user_group from source id's policy, making it public again
        db.delete_source_id_read_access_policy(source_id, user_group_id)
            .await
            .unwrap();

        // user1 and user2 can access source again
        assert!(policy1.check_read_source(source_id).await.is_ok());
        assert!(policy2.check_read_source(source_id).await.is_ok());
    }
}
