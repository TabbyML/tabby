use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

use super::{interface::UserValue, Context};
use crate::Result;

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct UserGroup {
    pub id: ID,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub members: Vec<UserGroupMembership>,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct UserGroupMembership {
    pub user: UserValue,

    pub is_group_admin: bool,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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
pub struct UpsertUserGroupMembershipInput {
    pub user_group_id: ID,
    pub user_id: ID,
    pub is_group_admin: bool,
}

#[async_trait]
pub trait UserGroupService: Send + Sync {
    // List user groups.
    async fn list(&self) -> Result<Vec<UserGroup>>;

    async fn create(&self, input: &CreateUserGroupInput) -> Result<ID>;
    async fn delete(&self, user_group_id: &ID) -> Result<()>;

    async fn upsert_membership(&self, input: &UpsertUserGroupMembershipInput) -> Result<()>;
    async fn delete_membership(&self, user_group_id: &ID, user_id: &ID) -> Result<()>;
}
