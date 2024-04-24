use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tabby_db::DbConn;
use juniper::ID;
use tracing::warn;

use super::{graphql_pagination_to_filter, AsRowid};
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
        users: Vec<ID>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<UserEvent>> {
        let users = convert_ids(users);
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let events = self
            .db
            .list_user_events(limit, skip_id, backwards, users, start.into(), end.into())
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