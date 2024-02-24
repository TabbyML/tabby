use std::error::Error;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject};
use serde::Deserialize;

use crate::schema::Result;

use super::CoreError;

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

impl Into<CoreError> for LicenseStatus {
    fn into(self) -> CoreError {
        match self {
            LicenseStatus::Ok => panic!("License is valid, shouldn't be converted to CoreError"),
            LicenseStatus::Expired => CoreError::InvalidLicense("Your enterprise license is expired"),
            LicenseStatus::SeatsExceeded => CoreError::InvalidLicense("You have more active users than seats included in your license"),
        }
    }
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

impl LicenseInfo {
    pub fn seat_limits_for_community_license() -> usize {
        5
    }

    pub fn seat_limits_for_team_license() -> usize {
        30
    }

    pub fn check_node_limit(&self, num_nodes: usize) -> bool {
        match self.r#type {
            LicenseType::Community => false,
            LicenseType::Team => num_nodes < 2,
            LicenseType::Enterprise => true
        }
    }

    pub fn ensure_seat_limit(mut self) -> Self {
        let seats = self.seats as usize;
        self.seats = match self.r#type {
            LicenseType::Community => std::cmp::max(seats, Self::seat_limits_for_community_license()),
            LicenseType::Team => std::cmp::max(seats, Self::seat_limits_for_team_license()),
            LicenseType::Enterprise => seats
        } as i32;

        self
    }
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
