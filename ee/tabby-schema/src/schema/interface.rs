use chrono::{DateTime, Utc};
use juniper::GraphQLInterface;

use super::{auth::User, Context};
use crate::juniper::relay;

#[derive(GraphQLInterface)]
#[graphql(for = User, context = Context)]
pub struct UserInfo {
    pub id: juniper::ID,
    pub email: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

impl relay::NodeType for UserInfoValue {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        match self {
            UserInfoValueEnum::User(user) => user.id.to_string(),
        }
    }

    fn connection_type_name() -> &'static str {
        "UserInfoConnection"
    }

    fn edge_type_name() -> &'static str {
        "UserInfoEdge"
    }
}
