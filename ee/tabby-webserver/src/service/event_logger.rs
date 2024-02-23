use std::fmt::Display;

use tabby_common::api::event::EventLogger;
use tabby_db::DbConn;

struct EventLoggerImpl {
    db: DbConn,
}

fn log_err<T, E: Display>(res: Result<T, E>) {
    if let Err(e) = res {
        eprintln!("Failed to log event: {e}");
    }
}

pub fn new_event_logger(db: DbConn) -> impl EventLogger {
    EventLoggerImpl { db }
}

impl EventLogger for EventLoggerImpl {
    fn log(&self, e: tabby_common::api::event::Event) {
        match e {
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
                    let user_db = db.get_user_by_email(&user).await;
                    let Ok(Some(user_db)) = user_db else {
                        eprintln!("Failed to retrieve user for {user}");
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

    use super::*;
    use tabby_common::api::event::Event;
    use tabby_db::DbConn;

    async fn sleep_50() {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn test_event_logger() {
        let db = DbConn::new_in_memory().await.unwrap();
        let logger = new_event_logger(db.clone());
        logger.log(Event::Completion {
            completion_id: "test_id".into(),
            language: "rust".into(),
            prompt: "testprompt".into(),
            segments: None,
            choices: vec![],
            user: Some("example".into()),
        });

        sleep_50().await;
        assert!(!db.list_user_completions().await.unwrap().is_empty());

        logger.log(Event::View {
            completion_id: "test_id".into(),
            choice_index: 0,
            view_id: None,
        });

        sleep_50().await;
        assert_eq!(!db.list_user_completions().await.unwrap()[0].views, 1);

        logger.log(Event::Dismiss {
            completion_id: "test_id".into(),
            choice_index: 0,
            view_id: None,
            elapsed: None,
        });

        sleep_50().await;
        assert_eq!(!db.list_user_completions().await.unwrap()[0].dismisses, 1);

        logger.log(Event::Select {
            completion_id: "test_id".into(),
            choice_index: 0,
            view_id: None,
            kind: None,
            elapsed: None,
        });

        sleep_50().await;
        assert_eq!(!db.list_user_completions().await.unwrap()[0].selects, 1);
    }
}
