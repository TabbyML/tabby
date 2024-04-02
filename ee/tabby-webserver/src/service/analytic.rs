use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;

use crate::schema::analytic::{AnalyticService, CompletionStats, Language};
use crate::schema::Result;

struct AnalyticServiceImpl {}

#[async_trait]
impl AnalyticService for AnalyticServiceImpl {
    async fn annual_activity(&self) -> Result<Vec<CompletionStats>> {
        todo!()
    }

    async fn daily_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<ID>,
        languages: Vec<Language>,
    ) -> Result<CompletionStats> {
        todo!()
    }
}

pub fn new_analytic_service() -> Arc<dyn AnalyticService> {
    Arc::new(AnalyticServiceImpl {})
}