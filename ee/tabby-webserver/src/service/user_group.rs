use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    user_group::{
        CreateUserGroupInput, UpsertUserGroupMembershipInput, UserGroup, UserGroupService,
    },
    AsID, AsRowid, Result,
};

use super::{UserGroupExt, UserSecuredExt};

struct UserGroupServiceImpl {
    db: DbConn,
}

#[async_trait::async_trait]
impl UserGroupService for UserGroupServiceImpl {
    async fn list(&self) -> Result<Vec<UserGroup>> {
        let mut user_groups = Vec::new();
        for x in self.db.list_user_groups(None).await? {
            user_groups.push(UserGroup::new(self.db.clone(), x).await?);
        }
        Ok(user_groups)
    }

    async fn create(&self, input: &CreateUserGroupInput) -> Result<ID> {
        let id = self.db.create_user_group(&input.name).await?;
        Ok(id.as_id())
    }

    async fn delete(&self, user_group_id: &ID) -> Result<()> {
        self.db.delete_user_group(user_group_id.as_rowid()?).await?;
        Ok(())
    }

    async fn upsert_membership(&self, input: &UpsertUserGroupMembershipInput) -> Result<()> {
        self.db
            .upsert_user_group_membership(
                input.user_id.as_rowid()?,
                input.user_group_id.as_rowid()?,
                input.is_group_admin,
            )
            .await?;
        Ok(())
    }
    async fn delete_membership(&self, user_group_id: &ID, user_id: &ID) -> Result<()> {
        self.db
            .delete_user_group_membership(user_id.as_rowid()?, user_group_id.as_rowid()?)
            .await?;
        Ok(())
    }
}

pub fn create(db: DbConn) -> impl UserGroupService {
    UserGroupServiceImpl { db }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use tabby_db::testutils;
    use tabby_schema::interface::UserValue;

    use super::*;

    #[tokio::test]
    async fn test_list_user_groups() {
        // Create an in-memory database connection
        let db = DbConn::new_in_memory().await.unwrap();
        let svc = create(db.clone());

        // Insert test users into the database
        let user1 = testutils::create_user(&db).await.as_id();
        let user2 = testutils::create_user2(&db).await.as_id();

        // Insert test user groups associated with the users
        let user_group1 = svc
            .create(&CreateUserGroupInput {
                name: "group1".to_owned(),
            })
            .await
            .unwrap();
        let user_group2 = svc
            .create(&CreateUserGroupInput {
                name: "group2".to_owned(),
            })
            .await
            .unwrap();

        // Add user1 / user2 to user_group1
        svc.upsert_membership(&UpsertUserGroupMembershipInput {
            user_group_id: user_group1.clone(),
            user_id: user1.clone(),
            is_group_admin: true,
        })
        .await
        .unwrap();

        svc.upsert_membership(&UpsertUserGroupMembershipInput {
            user_group_id: user_group1.clone(),
            user_id: user2.clone(),
            is_group_admin: false,
        })
        .await
        .unwrap();

        // Add user2 to user_group2
        svc.upsert_membership(&UpsertUserGroupMembershipInput {
            user_group_id: user_group2.clone(),
            user_id: user2.clone(),
            is_group_admin: false,
        })
        .await
        .unwrap();

        // Test listing user groups as user3
        let result = svc.list().await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, user_group1);
        assert_eq!(result[1].id, user_group2);

        // Test listing user groups as user1
        let result = svc.list().await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, user_group1);
        assert_eq!(result[0].members.len(), 2);
        assert_matches!(&result[0].members[0].user, UserValue::UserSecured(user) => {
            assert_eq!(user.id, user1);
        });
        assert_matches!(&result[0].members[1].user, UserValue::UserSecured(user) => {
            assert_eq!(user.id, user2);
        });

        // Test listing user groups as user2
        let result = svc.list().await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, user_group1);
        assert_eq!(result[1].id, user_group2);
        assert_eq!(result[1].members.len(), 1);
        assert_matches!(&result[1].members[0].user, UserValue::UserSecured(user) => {
            assert_eq!(user.id, user2);
        });
    }
}
