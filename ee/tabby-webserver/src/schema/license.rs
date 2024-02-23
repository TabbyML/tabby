use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject};
use serde::Deserialize;

use crate::schema::Result;

#[derive(Debug, Deserialize, GraphQLEnum)]
#[serde(rename_all = "UPPERCASE")]
pub enum LicenseType {
    Team,
}

#[derive(GraphQLEnum, PartialEq, Debug)]
pub enum LicenseStatus {
    Ok,
    Expired,
    SeatsExceeded,
}

#[derive(GraphQLObject)]
pub struct LicenseInfo {
    pub r#type: LicenseType,
    pub status: LicenseStatus,
    pub seats: i32,
    pub seats_used: i32,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

#[async_trait]
pub trait LicenseService: Send + Sync {
    async fn read_license(&self) -> Result<Option<LicenseInfo>>;
    async fn update_license(&self, license: Option<String>) -> Result<Option<LicenseStatus>>;
}
