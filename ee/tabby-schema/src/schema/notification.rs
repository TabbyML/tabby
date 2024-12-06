use juniper::GraphQLEnum;

#[derive(GraphQLEnum, Clone, Debug)]
pub enum NotificationRecipient {
    Admin,
    AllUser,
}
