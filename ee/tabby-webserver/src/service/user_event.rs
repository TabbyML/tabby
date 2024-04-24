use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tabby_db::DbConn;

use super::graphql_pagination_to_filter;
use crate::schema::{
    user_event::{UserEvent, UserEventService},
    Result,
};

struct UserEventServiceImpl {
    db: DbConn,
}

pub fn create(db: DbConn) -> impl UserEventService {
    UserEventServiceImpl { db }
}

#[async_trait]
impl UserEventService for UserEventServiceImpl {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<UserEvent>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let events = self
            .db
            .list_user_events(limit, skip_id, backwards, start, end)
            .await?;
        Ok(events
            .into_iter()
            .map(UserEvent::try_from)
            .collect::<Result<_, _>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use tabby_db::DbConn;

    fn timestamp() -> u128 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let start = SystemTime::now();
        start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis()
    }

    #[tokio::test]
    async fn test_list_events() {
        let db = DbConn::new_in_memory().await.unwrap();

        let user = db.create_user("testuser".into(), None, true).await.unwrap();
        db.create_user_event(
            user,
            "completion".into(),
            timestamp(),
            "event payload".into(),
        )
        .await
        .unwrap();

        let service = create(db.clone());

        assert_eq!(
            1,
            service
                .list(
                    None,
                    None,
                    None,
                    None,
                    Utc::now() - Duration::minutes(1),
                    Utc::now() + Duration::minutes(1),
                )
                .await
                .unwrap()
                .len()
        );
    }
}
