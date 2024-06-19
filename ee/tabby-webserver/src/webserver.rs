use std::sync::Arc;

use axum::Router;
use tabby_common::{
    api::{
        code::CodeSearch,
        event::{ComposedLogger, EventLogger},
    },
    config::{Config, ConfigAccess, RepositoryConfig},
};
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{
    integration::IntegrationService, repository::RepositoryService, web_crawler::WebCrawlerService,
};

use crate::{
    path::db_file,
    routes,
    service::{
        background_job::{self, BackgroundJobEvent},
        create_service_locator,
        event_logger::create_event_logger,
        integration, repository, web_crawler,
    },
};

pub struct Webserver {
    db: DbConn,
    logger: Arc<dyn EventLogger>,
    repository: Arc<dyn RepositoryService>,
    integration: Arc<dyn IntegrationService>,
    web_crawler: Arc<dyn WebCrawlerService>,
    background_job_sender: tokio::sync::mpsc::UnboundedSender<BackgroundJobEvent>,
}

#[async_trait::async_trait]
impl ConfigAccess for Webserver {
    async fn repositories(&self) -> anyhow::Result<Vec<RepositoryConfig>> {
        let mut repos = Config::load().map(|x| x.repositories).unwrap_or_default();
        repos.extend(self.repository.list_all_repository_urls().await?);
        Ok(repos)
    }
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

        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel::<BackgroundJobEvent>();

        let integration = Arc::new(integration::create(db.clone(), sender.clone()));
        let repository = repository::create(db.clone(), integration.clone(), sender.clone());

        let web_crawler = Arc::new(web_crawler::create(db.clone(), sender.clone()));

        let logger2 = create_event_logger(db.clone());
        let logger = Arc::new(ComposedLogger::new(logger1, logger2));
        let ws = Arc::new(Webserver {
            db: db.clone(),
            logger,
            repository: repository.clone(),
            integration: integration.clone(),
            web_crawler: web_crawler.clone(),
            background_job_sender: sender.clone(),
        });

        background_job::start(
            db.clone(),
            repository.git(),
            repository.third_party(),
            integration.clone(),
            web_crawler,
            embedding,
            sender,
            receiver,
        )
        .await;

        ws
    }

    pub fn logger(&self) -> Arc<dyn EventLogger + 'static> {
        self.logger.clone()
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
            self.integration.clone(),
            self.web_crawler.clone(),
            self.db.clone(),
            self.background_job_sender.clone(),
            is_chat_enabled,
        )
        .await;

        routes::create(ctx, api, ui)
    }
}
