use crate::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

#[derive(GraphQLObject)]
pub struct UserGroup {
    id: ID,
    name: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,

    members: Vec<UserGroupMember>,
}

#[derive(GraphQLObject)]
pub struct UserGroupMember {
    user_id: ID,

    is_group_admin: bool,

    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateUserGroupInput {
    /// User group name, can only start with a lowercase letter, and contain characters, numbers, and `-` or `_`
    #[validate(length(
        min = 2,
        max = 20,
        code = "name",
        message = "Name must be between 2 and 20 characters"
    ))]
    #[validate(regex(
        code = "name",
        path = "*crate::schema::constants::USER_GROUP_NAME_REGEX",
        message = "Invalid name, name may contain characters which are not supported"
    ))]
    pub name: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct  UpsertUserGroupMembershipInput {
     pub user_group_id: ID,
     pub user_id: ID,
     pub is_group_admin: bool,
}

#[async_trait]
pub trait UserGroupService: Send + Sync {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<UserGroup>>;

    async fn create(&self, input: &CreateUserGroupInput) -> Result<ID>;
    async fn delete(&self, user_group_id: &ID) -> Result<()>;

    async fn upsert_membership(&self, input: &UpsertUserGroupMembershipInput) -> Result<()>;
    async fn delete_membership(&self, user_group_id: &ID, user_id: &ID) -> Result<()>;
}
