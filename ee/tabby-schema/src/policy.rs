use juniper::ID;
use tabby_db::DbConn;

use crate::{AsRowid, CoreError, Result};

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
        let allow = self.db.allow_read_source(self.user_id.as_rowid()?, source_id).await?;
        if !allow {
            return Err(CoreError::Forbidden(
                "You are not allowed to read this source",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::testutils;

    use crate::AsID;

    use super::*;

    async fn testdb() -> DbConn {
        DbConn::new_in_memory().await.unwrap()
    }

    #[tokio::test]
    async fn test_check_read_source() {
        let db = testdb().await;
        let user_id1 = testutils::create_user(&db).await;
        let user_id2 = testutils::create_user2(&db).await;

        let policy1 = AccessPolicy::new(db.clone(), &user_id1.as_id(), false);
        let policy2 = AccessPolicy::new(db.clone(), &user_id2.as_id(), false);

        // 1. Setup user group
        let user_group_id = db.create_user_group("test").await.unwrap();
        db.upsert_user_group_membership(user_id1, user_group_id, false).await.unwrap();

        // For source id without any access policies, it's public (readable by all users)
        assert!(policy1.check_read_source("source").await.is_ok());
        assert!(policy2.check_read_source("source").await.is_ok());

        // 2. add user_group to source id's policy, making it private
        db.upsert_source_id_read_access_policy("source", user_group_id).await.unwrap();

        // user2 won't be able to access source, while user1 can.
        assert!(policy2.check_read_source("source").await.is_err());
        assert!(policy1.check_read_source("source").await.is_ok());

        // 3. remove user1 from user_group
        db.delete_user_group_membership(user_id1, user_group_id).await.unwrap();

        // user1 won't be able to acces source either now.
        assert!(policy1.check_read_source("source").await.is_err());

    }
}
