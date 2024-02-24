use std::error::Error;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject};
use serde::Deserialize;

use crate::schema::Result;

#[derive(Debug, Deserialize, GraphQLEnum, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum LicenseType {
    Community,
    Team,
    Enterprise,
}

#[derive(GraphQLEnum, PartialEq, Debug, Clone)]
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
    pub issued_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[async_trait]
pub trait LicenseService: Send + Sync {
    async fn read_license(&self) -> Result<LicenseInfo>;
    async fn update_license(&self, license: String) -> Result<()>;
}

pub trait IsLicenseValid {
    fn is_license_valid(&self) -> bool;
}

impl IsLicenseValid for LicenseInfo {
    fn is_license_valid(&self) -> bool {
        self.status == LicenseStatus::Ok
    }
}

impl<L: IsLicenseValid, T: Error> IsLicenseValid for std::result::Result<L, T> {
    fn is_license_valid(&self) -> bool {
        if let Ok(x) = self {
            x.is_license_valid()
        } else {
            false
        }
    }
}
