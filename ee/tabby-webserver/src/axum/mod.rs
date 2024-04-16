pub mod extract;

use axum::{
    extract::{Extension, State},
    response::IntoResponse,
};
use extract::AuthBearer;
use juniper_axum::{extract::JuniperRequest, response::JuniperResponse};
use juniper_graphql_ws::Schema;

pub trait FromAuth<S> {
    fn build(state: S, bearer: Option<String>) -> Self;
}

#[cfg_attr(text, axum::debug_handler)]
pub async fn graphql<S, C>(
    State(state): State<C>,
    Extension(schema): Extension<S>,
    AuthBearer(bearer): AuthBearer,
    JuniperRequest(req): JuniperRequest<S::ScalarValue>,
) -> impl IntoResponse
where
    S: Schema, // TODO: Refactor in the way we don't depend on `juniper_graphql_ws::Schema` here.
    S::Context: FromAuth<C>,
{
    let ctx = S::Context::build(state, bearer);
    JuniperResponse(req.execute(schema.root_node(), &ctx).await).into_response()
}
