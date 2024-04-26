use juniper::{GraphQLInputObject, ID};
use validator::Validate;

#[derive(GraphQLInputObject, Validate)]
pub struct CreateRepositoryProviderInput {
    #[validate(regex(
        code = "displayName",
        path = "crate::schema::constants::REPOSITORY_NAME_REGEX"
    ))]
    pub display_name: String,
    #[validate(length(code = "access_token", min = 10))]
    pub access_token: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdateRepositoryProviderInput {
    pub id: ID,
    #[validate(regex(
        code = "displayName",
        path = "crate::schema::constants::REPOSITORY_NAME_REGEX"
    ))]
    pub display_name: String,
    #[validate(length(code = "access_token", min = 10))]
    pub access_token: String,
}
