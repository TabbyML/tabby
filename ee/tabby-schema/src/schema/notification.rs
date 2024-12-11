use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};

#[derive(GraphQLEnum, Clone, Debug)]
pub enum NotificationRecipient {
    Admin,
    AllUser,
}

#[derive(GraphQLObject)]
pub struct Notification {
    pub id: ID,
    pub content: String,
    pub read: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
