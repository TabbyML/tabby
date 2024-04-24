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
            .list_user_events(limit, skip_id, backwards, start.into(), end.into())
            .await?;
        Ok(events
            .into_iter()
            .map(UserEvent::try_from)
            .collect::<Result<_, _>>()?)
    }
}
