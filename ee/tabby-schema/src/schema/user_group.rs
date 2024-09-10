use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

use crate::{policy::AccessPolicy, Result};

#[derive(GraphQLObject)]
pub struct UserGroup {
    pub id: ID,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(GraphQLObject)]
pub struct UserGroupMembership {
    pub id: ID,
    pub user_group_id: ID,
    pub user_id: ID,

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
    //
    // * When user is admin, returns all user groups.
    // * Otherwise, returns user groups that the user is a member of.
    async fn list(&self, policy: &AccessPolicy) -> Result<Vec<UserGroup>>;

    async fn create(&self, input: &CreateUserGroupInput) -> Result<ID>;
    async fn delete(&self, user_group_id: &ID) -> Result<()>;

    // List user group memberships.
    //
    // * When user_id is provided, it acts as a filter, returning either 1 or 0 results.
    async fn list_membership(
        &self,
        policy: &AccessPolicy,
        user_group_id: &ID,
    ) -> Result<Vec<UserGroupMembership>>;

    async fn upsert_membership(&self, input: &UpsertUserGroupMembershipInput) -> Result<()>;
    async fn delete_membership(&self, user_group_id: &ID, user_id: &ID) -> Result<()>;
}
