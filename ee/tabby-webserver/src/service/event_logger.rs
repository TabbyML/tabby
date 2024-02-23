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
