use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::Response,
};
use hyper::{header::CONTENT_TYPE, Body, StatusCode};
use juniper::ID;
use tracing::error;

use crate::schema::auth::AuthenticationService;

pub async fn avatar(
    State(state): State<Arc<dyn AuthenticationService>>,
    Path(id): Path<ID>,
) -> Result<Response<Body>, StatusCode> {
    let avatar = state
        .get_user_avatar(&id)
        .await
        .map_err(|e| {
            error!("Failed to retrieve avatar: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or(StatusCode::NOT_FOUND)?;
    let mut response = Response::new(Body::from(avatar.into_vec()));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, "image/*".parse().unwrap());
    Ok(response)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::extract::{Path, State};
    use hyper::body::to_bytes;

    use super::avatar;
    use crate::{
        auth::new_authentication_service, email::new_email_service, license::new_license_service,
        schema::auth::AuthenticationService, AsID,
    };

    #[tokio::test]
    async fn test_fetch_avatar() {
        use tabby_db::DbConn;

        let db = DbConn::new_in_memory().await.unwrap();

        let auth = Arc::new(new_authentication_service(
            db.clone(),
            Arc::new(new_email_service(db.clone()).await.unwrap()),
            Arc::new(new_license_service(db.clone()).await.unwrap()),
        ));

        let id = db
            .create_user("test@example.com".into(), None, true)
            .await
            .unwrap()
            .as_id();

        let test_avatar = "test";
        auth.update_user_avatar(
            &id,
            Some(test_avatar.as_bytes().to_vec().into_boxed_slice()),
        )
        .await
        .unwrap();
        let avatar = avatar(State(auth), Path(id)).await.unwrap().into_body();
        let avatar = to_bytes(avatar).await.unwrap();
        assert_eq!(avatar, test_avatar.as_bytes());
    }
}
