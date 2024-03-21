use std::fmt::Display;

use juniper::ID;
use tabby_common::api::event::{Log, RawEventLogger};
use tabby_db::DbConn;
use tracing::warn;

use super::dao::AsRowid;

struct DbEventLogger {
    db: DbConn,
}

fn log_err<T, E: Display>(res: Result<T, E>) {
    if let Err(e) = res {
        warn!("Failed to log event: {e}");
    }
}

pub fn new_event_logger(db: DbConn) -> impl RawEventLogger {
    DbEventLogger { db }
}

impl RawEventLogger for DbEventLogger {
    fn log_raw(&self, content: String) {
        let Ok(Log { event, .. }) = serde_json::from_str(&content) else {
            warn!("Invalid event JSON: {content}");
            return;
        };
        match event {
            tabby_common::api::event::Event::View { completion_id, .. } => {
                let db = self.db.clone();
                tokio::spawn(async move {
                    log_err(db.add_to_user_completion(&completion_id, 1, 0, 0).await)
                });
            }
            tabby_common::api::event::Event::Select { completion_id, .. } => {
                let db = self.db.clone();
                tokio::spawn(async move {
                    log_err(db.add_to_user_completion(&completion_id, 0, 1, 0).await)
                });
            }
            tabby_common::api::event::Event::Dismiss { completion_id, .. } => {
                let db = self.db.clone();
                tokio::spawn(async move {
                    log_err(db.add_to_user_completion(&completion_id, 0, 0, 1).await)
                });
            }
            tabby_common::api::event::Event::Completion {
                completion_id,
                language,
                user,
                ..
            } => {
                let Some(user) = user else { return };
                let db = self.db.clone();
                tokio::spawn(async move {
                    let Ok(id) = ID::new(&user).as_rowid() else {
                        warn!("Invalid user ID");
                        return;
                    };
                    let user_db = db.get_user(id).await;
                    let Ok(Some(user_db)) = user_db else {
                        warn!("Failed to retrieve user for {user}");
                        return;
                    };
                    log_err(
                        db.create_user_completion(user_db.id, completion_id, language)
                            .await,
                    );
                });
            }
            tabby_common::api::event::Event::ChatCompletion { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tabby_common::api::event::{Event, Message, RawEventLogger};
    use tabby_db::DbConn;

    use crate::service::{dao::AsID, event_logger::new_event_logger};

    async fn sleep_50() {
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    #[tokio::test]
    async fn test_event_logger() {
        let db = DbConn::new_in_memory().await.unwrap();
        let logger = new_event_logger(db.clone());
        let user_id = db
            .create_user("testuser".into(), "pass".into(), true)
            .await
            .unwrap();

        let id = user_id.as_id();

        logger.log(Event::Completion {
            completion_id: "test_id".into(),
            language: "rust".into(),
            prompt: "testprompt".into(),
            segments: None,
            choices: vec![],
            user: Some(id.to_string()),
        });

        sleep_50().await;
        assert!(db.fetch_one_user_completion().await.unwrap().is_some());

        logger.log(Event::View {
            completion_id: "test_id".into(),
            choice_index: 0,
            view_id: None,
        });

        sleep_50().await;
        assert_eq!(
            db.fetch_one_user_completion().await.unwrap().unwrap().views,
            1
        );

        logger.log(Event::Dismiss {
            completion_id: "test_id".into(),
            choice_index: 0,
            view_id: None,
            elapsed: None,
        });

        sleep_50().await;
        assert_eq!(
            db.fetch_one_user_completion()
                .await
                .unwrap()
                .unwrap()
                .dismisses,
            1
        );

        logger.log(Event::Select {
            completion_id: "test_id".into(),
            choice_index: 0,
            view_id: None,
            kind: None,
            elapsed: None,
        });

        sleep_50().await;
        assert_eq!(
            db.fetch_one_user_completion()
                .await
                .unwrap()
                .unwrap()
                .selects,
            1
        );
    }

    #[tokio::test]
    async fn test_event_without_user_will_be_skipped() {
        let db = DbConn::new_in_memory().await.unwrap();
        let logger = new_event_logger(db.clone());

        logger.log(Event::Completion {
            completion_id: "test_id".into(),
            language: "rust".into(),
            prompt: "testprompt".into(),
            segments: None,
            choices: vec![],
            user: Some("testuser".into()),
        });

        sleep_50().await;
        assert!(db.fetch_one_user_completion().await.unwrap().is_none());

        logger.log(Event::Completion {
            completion_id: "test_id".into(),
            language: "rust".into(),
            prompt: "testprompt".into(),
            segments: None,
            choices: vec![],
            user: None,
        });

        sleep_50().await;
        assert!(db.fetch_one_user_completion().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_chat_completion_event_will_be_skipped() {
        let db = DbConn::new_in_memory().await.unwrap();
        let logger = new_event_logger(db.clone());

        logger.log(Event::ChatCompletion {
            completion_id: "test_id".into(),
            input: vec![],
            output: Message {
                role: "user".into(),
                content: "test".into(),
            },
        });

        sleep_50().await;
        assert!(db.fetch_one_user_completion().await.unwrap().is_none());
    }
}
