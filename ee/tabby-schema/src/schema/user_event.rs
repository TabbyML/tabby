use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};

use super::Context;
use crate::{juniper::relay::NodeType, schema::Result};

#[derive(GraphQLEnum, Debug)]
pub enum EventKind {
    Completion,
    ChatCompletion,
    Select,
    View,
    Dismiss,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct UserEvent {
    pub id: ID,
    pub user_id: ID,
    pub kind: EventKind,
    pub created_at: DateTime<Utc>,
    pub payload: String,
}

impl NodeType for UserEvent {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "UserEventConnection"
    }

    fn edge_type_name() -> &'static str {
        "UserEventEdge"
    }
}

#[async_trait]
pub trait UserEventService: Send + Sync {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
        users: Vec<ID>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<UserEvent>>;
}
