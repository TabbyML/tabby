use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    user_event::{UserEvent, UserEventService},
    AsRowid, Result,
};
use tracing::warn;

use super::graphql_pagination_to_filter;

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
        users: Vec<ID>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<UserEvent>> {
        let users = convert_ids(users);
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let events = self
            .db
            .list_user_events(limit, skip_id, backwards, users, start, end)
            .await?;
        Ok(events
            .into_iter()
            .map(UserEvent::try_from)
            .collect::<Result<_, _>>()?)
    }
}

fn convert_ids(ids: Vec<ID>) -> Vec<i64> {
    ids.into_iter()
        .filter_map(|id| match id.as_rowid() {
            Ok(rowid) => Some(rowid),
            Err(_) => {
                warn!("Ignoring invalid ID: {}", id);
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use chrono::{Days, Duration};
    use tabby_schema::{user_event::EventKind, AsID};

    use super::*;

    fn timestamp() -> u128 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let start = SystemTime::now();
        start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis()
    }

    #[tokio::test]
    async fn test_list_user_events() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user1 = db
            .create_user("test@example.com".into(), Some("pass".into()), true, None)
            .await
            .unwrap();

        db.create_user_event(user1, "view".into(), timestamp(), "".into())
            .await
            .unwrap();

        let user2 = db
            .create_user("test2@example.com".into(), Some("pass".into()), true, None)
            .await
            .unwrap();

        db.create_user_event(user2, "select".into(), timestamp(), "".into())
            .await
            .unwrap();

        let svc = create(db);
        let end = Utc::now() + Duration::days(1);
        let start = end.checked_sub_days(Days::new(100)).unwrap();

        // List without users should return all events
        let events = svc
            .list(None, None, None, None, vec![], start, end)
            .await
            .unwrap();
        assert_eq!(events.len(), 2);

        // Filter with user should return only events for that user
        let events = svc
            .list(None, None, None, None, vec![user1.as_id()], start, end)
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_matches!(events[0].kind, EventKind::View);
    }
}
