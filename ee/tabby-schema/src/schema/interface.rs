use chrono::{DateTime, Utc};
use juniper::GraphQLInterface;

use super::{auth::UserSecured, Context};
use crate::juniper::relay;

#[derive(GraphQLInterface)]
#[graphql(for = UserSecured, context = Context)]
pub struct User {
    pub id: juniper::ID,
    pub email: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub is_admin: bool,
    pub is_owner: bool,
    pub active: bool,
}

impl relay::NodeType for UserValue {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        match self {
            UserValueEnum::UserSecured(user) => user.id.to_string(),
        }
    }

    fn connection_type_name() -> &'static str {
        "UserConnection"
    }

    fn edge_type_name() -> &'static str {
        "UserEdge"
    }
}
