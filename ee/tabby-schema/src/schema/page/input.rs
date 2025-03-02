use juniper::GraphQLInputObject;

#[derive(GraphQLInputObject)]
pub struct CreatePageRunInput {
    pub title: String,
}
