use juniper::{GraphQLEnum, GraphQLInputObject, ID};
use validator::Validate;

use crate::integration::IntegrationKind;

#[derive(GraphQLInputObject, Validate)]
pub struct CreateIntegrationInput {
    #[validate(regex(
        code = "displayName",
        path = "*crate::schema::constants::REPOSITORY_NAME_REGEX",
        message = "Invalid repository provider name"
    ))]
    pub display_name: String,
    #[validate(length(code = "accessToken", min = 10, message = "Invalid access token"))]
    pub access_token: String,
    pub kind: IntegrationKind,
    #[validate(url(code = "apiBase", message = "Invalid URL base"))]
    pub api_base: Option<String>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateSelfHostedRepositoryProviderInput {
    #[validate(regex(
        code = "displayName",
        path = "*crate::schema::constants::REPOSITORY_NAME_REGEX",
        message = "Invalid repository provider name"
    ))]
    pub display_name: String,
    #[validate(length(code = "accessToken", min = 10, message = "Invalid access token"))]
    pub access_token: String,
    #[validate(url(code = "apiBase", message = "Invalid URL base"))]
    pub api_base: Option<String>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdateIntegrationInput {
    pub id: ID,
    #[validate(regex(
        code = "displayName",
        path = "*crate::schema::constants::REPOSITORY_NAME_REGEX",
        message = "Invalid repository provider name"
    ))]
    pub display_name: String,
    #[validate(length(code = "accessToken", min = 10, message = "Invalid access token"))]
    pub access_token: Option<String>,
    #[validate(url(code = "apiBase", message = "Invalid URL base"))]
    pub api_base: Option<String>,
    pub kind: IntegrationKind,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdateSelfHostedRepositoryProviderInput {
    pub id: ID,
    #[validate(regex(
        code = "displayName",
        path = "*crate::schema::constants::REPOSITORY_NAME_REGEX",
        message = "Invalid repository provider name"
    ))]
    pub display_name: String,
    #[validate(length(code = "accessToken", min = 10, message = "Invalid access token"))]
    pub access_token: Option<String>,
    #[validate(url(code = "apiBase", message = "Invalid URL base"))]
    pub api_base: Option<String>,
}

#[derive(GraphQLEnum, Debug, PartialEq)]
pub enum RepositoryProviderStatus {
    Ready,
    Pending,
    Failed,
}
