use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tabby_schema::{license::LicenseService, notification::NotificationService};

use super::helper::Job;
use crate::service::background_job::LicenseCheckJob;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DailyJob;

impl Job for DailyJob {
    const NAME: &'static str = "daily";
}

impl DailyJob {
    pub async fn run(
        &self,
        license_service: Arc<dyn LicenseService>,
        notification_service: Arc<dyn NotificationService>,
    ) -> tabby_schema::Result<()> {
        let now = Utc::now();

        if let Err(err) =
            LicenseCheckJob::cron(now, license_service.clone(), notification_service.clone()).await
        {
            logkit::warn!("License check job failed: {err:?}");
        }
        Ok(())
    }
}
