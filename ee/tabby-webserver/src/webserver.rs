use std::sync::Arc;

use axum::Router;
use tabby_common::{
    api::{
        code::CodeSearch,
        doc::DocSearch,
        event::{ComposedLogger, EventLogger},
    },
    config::Config,
};
use tabby_db::DbConn;
use tabby_inference::{ChatCompletionStream, Embedding};
use tabby_schema::job::JobService;
use tracing::debug;

use crate::{
    path::db_file,
    routes,
    service::{
        create_service_locator, event_logger::create_event_logger, integration, job, repository,
        web_documents,
    },
};

pub struct Webserver {
    db: DbConn,
    logger: Arc<dyn EventLogger>,
    embedding: Arc<dyn Embedding>,
}

impl Webserver {
    pub async fn new(
        logger1: impl EventLogger + 'static,
        embedding: Arc<dyn Embedding>,
    ) -> Arc<Self> {
        let db = DbConn::new(db_file().as_path())
            .await
            .expect("Must be able to initialize db");
        db.finalize_stale_job_runs()
            .await
            .expect("Must be able to finalize stale job runs");

        let logger2 = create_event_logger(db.clone());
        let logger = Arc::new(ComposedLogger::new(logger1, logger2));

        Arc::new(Webserver {
            db: db.clone(),
            logger,
            embedding,
        })
    }

    pub fn logger(&self) -> Arc<dyn EventLogger + 'static> {
        self.logger.clone()
    }

    pub async fn attach(
        &self,
        config: &Config,
        api: Router,
        ui: Router,
        code: Arc<dyn CodeSearch>,
        chat: Option<Arc<dyn ChatCompletionStream>>,
        docsearch: Arc<dyn DocSearch>,
        serper_factory_fn: impl Fn(&str) -> Box<dyn DocSearch>,
    ) -> (Router, Router) {
        let serper: Option<Box<dyn DocSearch>> =
            if let Ok(api_key) = std::env::var("SERPER_API_KEY") {
                debug!("Serper API key found, enabling serper...");
                Some(serper_factory_fn(&api_key))
            } else {
                None
            };

        let db = self.db.clone();
        let job: Arc<dyn JobService> = Arc::new(job::create(db.clone()).await);

        let integration = Arc::new(integration::create(db.clone(), job.clone()));
        let repository = repository::create(db.clone(), integration.clone(), job.clone());

        let web_documents = Arc::new(web_documents::create(db.clone(), job.clone()));

        let context = Arc::new(crate::service::context::create(
            repository.clone(),
            web_documents.clone(),
            serper.is_some(),
        ));

        let answer = chat.as_ref().map(|chat| {
            Arc::new(crate::service::answer::create(
                &config.answer,
                chat.clone(),
                code.clone(),
                docsearch.clone(),
                context.clone(),
                serper,
            ))
        });

        let is_chat_enabled = chat.is_some();
        let ctx = create_service_locator(
            self.logger(),
            code.clone(),
            repository.clone(),
            integration.clone(),
            job.clone(),
            answer.clone(),
            context.clone(),
            web_documents.clone(),
            self.db.clone(),
            self.embedding.clone(),
            is_chat_enabled,
        )
        .await;

        routes::create(ctx, api, ui, answer)
    }
}
