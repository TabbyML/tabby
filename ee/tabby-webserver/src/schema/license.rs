use std::error::Error;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject};
use serde::Deserialize;

use super::CoreError;
use crate::schema::Result;

#[derive(Debug, Deserialize, GraphQLEnum, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum LicenseType {
    Community,
    Team,
    Enterprise,
    Demo,
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
            LicenseType::Team => num_nodes <= 2,
            LicenseType::Enterprise => true,
            LicenseType::Demo => false,
        }
    }

    pub fn guard_seat_limit(mut self) -> Self {
        let seats = self.seats as usize;
        self.seats = match self.r#type {
            LicenseType::Community => {
                std::cmp::min(seats, Self::seat_limits_for_community_license())
            }
            LicenseType::Team => std::cmp::min(seats, Self::seat_limits_for_team_license()),
            LicenseType::Enterprise => seats,
            LicenseType::Demo => usize::MAX,
        } as i32;

        self
    }

    pub fn ensure_available_seats(&self, num_new_seats: usize) -> Result<()> {
        self.ensure_valid_license()?;
        if (self.seats_used as usize + num_new_seats) > self.seats as usize {
            return Err(CoreError::InvalidLicense(
                "No sufficient seats under current license",
            ));
        }
        Ok(())
    }

    pub fn ensure_admin_seats(&self, num_admins: usize) -> Result<()> {
        self.ensure_valid_license()?;
        let num_admin_seats = match self.r#type {
            LicenseType::Community => 1,
            LicenseType::Team => 3,
            LicenseType::Enterprise => usize::MAX,
            LicenseType::Demo => usize::MAX,
        };

        if num_admins > num_admin_seats {
            return Err(CoreError::InvalidLicense(
                "No sufficient admin seats under the license",
            ));
        }

        Ok(())
    }
}

#[async_trait]
pub trait LicenseService: Send + Sync {
    async fn read_license(&self) -> Result<LicenseInfo>;
    async fn update_license(&self, license: String) -> Result<()>;
    async fn reset_license(&self) -> Result<()>;
}

pub trait IsLicenseValid {
    fn ensure_valid_license(&self) -> Result<()>;
}

impl IsLicenseValid for LicenseInfo {
    fn ensure_valid_license(&self) -> Result<()> {
        match self.status {
            LicenseStatus::Expired => Err(CoreError::InvalidLicense(
                "Your enterprise license is expired",
            )),
            LicenseStatus::SeatsExceeded => Err(CoreError::InvalidLicense(
                "You have more active users than seats included in your license",
            )),
            LicenseStatus::Ok => Ok(()),
        }
    }
}

impl<L: IsLicenseValid, T: Error> IsLicenseValid for std::result::Result<L, T> {
    fn ensure_valid_license(&self) -> Result<()> {
        if let Ok(x) = self {
            x.ensure_valid_license()
        } else {
            Err(CoreError::InvalidLicense("No valid license configured"))
        }
    }
}
