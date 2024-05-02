use std::sync::Arc;

use axum::Router;
use tabby_common::{
    api::{
        code::CodeSearch,
        event::{ComposedLogger, EventLogger},
    },
    config::RepositoryAccess,
};
use tabby_db::DbConn;
use tabby_schema::repository::RepositoryService;

use crate::{
    path::db_file,
    routes,
    service::{
        background_job, create_service_locator, event_logger::create_event_logger, repository,
    },
};

pub struct Webserver {
    db: DbConn,
    logger: Arc<dyn EventLogger>,
    repository: Arc<dyn RepositoryService>,
}

impl Webserver {
    pub async fn new(logger1: impl EventLogger + 'static, local_port: u16) -> Self {
        let db = DbConn::new(db_file().as_path())
            .await
            .expect("Must be able to initialize db");
        db.finalize_stale_job_runs()
            .await
            .expect("Must be able to finalize stale job runs");

        let background_job = background_job::create(db.clone(), local_port).await;
        let repository = repository::create(db.clone(), background_job);

        let logger2 = create_event_logger(db.clone());
        let logger = Arc::new(ComposedLogger::new(logger1, logger2));
        Webserver {
            db,
            logger,
            repository,
        }
    }

    pub fn logger(&self) -> Arc<dyn EventLogger + 'static> {
        self.logger.clone()
    }

    pub fn repository_access(&self) -> Arc<dyn RepositoryAccess> {
        self.repository.clone().access()
    }

    pub async fn attach(
        &self,
        api: Router,
        ui: Router,
        code: Arc<dyn CodeSearch>,
        is_chat_enabled: bool,
    ) -> (Router, Router) {
        let ctx = create_service_locator(
            self.logger(),
            code,
            self.repository.clone(),
            self.db.clone(),
            is_chat_enabled,
        )
        .await;

        routes::create(ctx, self.repository_access(), api, ui)
    }
}
