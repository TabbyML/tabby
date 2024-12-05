use juniper::GraphQLEnum;

#[derive(GraphQLEnum, Clone, Debug)]
pub enum NotificationKind {
    Admin,
    AllUser,
}
