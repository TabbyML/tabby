use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use jsonwebtoken as jwt;
use lazy_static::lazy_static;
use serde::Deserialize;
use std::sync::Arc;
use tabby_db::DbConn;
use tokio::sync::RwLock;

use crate::schema::license::{LicenseInfo, LicenseService, LicenseStatus, LicenseType};

lazy_static! {
    static ref LICENSE_DECODING_KEY: jwt::DecodingKey =
        jwt::DecodingKey::from_rsa_pem(include_bytes!("../../keys/license.key.pub")).unwrap();
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
struct LicenseJWTPayload {
    /// Expiration time (as UTC timestamp)
    pub exp: i64,

    /// Issued at (as UTC timestamp)
    pub iat: i64,

    /// Issuer
    pub iss: String,

    /// License grantee email address
    pub sub: String,

    /// License Type
    pub typ: LicenseType,

    /// Number of license (# of seats).
    pub num: usize,
}

fn validate_license(token: &str) -> Result<LicenseJWTPayload, jwt::errors::ErrorKind> {
    let mut validation = jwt::Validation::new(jwt::Algorithm::RS512);
    validation.validate_exp = false;
    validation.set_issuer(&["tabbyml.com"]);
    validation.set_required_spec_claims(&["exp", "iat", "sub", "iss"]);
    let data = jwt::decode::<LicenseJWTPayload>(token, &LICENSE_DECODING_KEY, &validation);
    let data = data.map_err(|err| match err.kind() {
        // Map json error (missing failed, parse error) as missing required claims.
        jwt::errors::ErrorKind::Json(err) => {
            jwt::errors::ErrorKind::MissingRequiredClaim(err.to_string())
        }
        _ => err.into_kind(),
    });
    Ok(data?.claims)
}

fn jwt_timestamp_to_utc(secs: i64) -> Result<DateTime<Utc>> {
    Ok(NaiveDateTime::from_timestamp_opt(secs, 0)
        .ok_or_else(|| anyhow!("Timestamp is corrupt"))?
        .and_utc())
}

struct LicenseServiceImpl {
    db: DbConn,
    seats: Arc<RwLock<(DateTime<Utc>, usize)>>,
}

impl LicenseServiceImpl {
    async fn read_used_seats(&self) -> Result<usize> {
        let now = Utc::now();
        let lock = self.seats.read().await;
        let (refreshed, seats) = &*lock;
        if now - refreshed > Duration::minutes(5) {
            let seats_clone = self.seats.clone();
            let db = self.db.clone();
            tokio::spawn(async move {
                let mut lock = seats_clone.write().await;
                let seats = match db.count_users().await {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("Failed to count users: {e}");
                        return;
                    }
                };
                *lock = (now, seats);
            });
        }
        Ok(*seats)
    }
}

pub fn new_license_service(db: DbConn) -> impl LicenseService {
    LicenseServiceImpl {
        db,
        seats: Default::default(),
    }
}

fn license_info_from_raw(raw: LicenseJWTPayload, seats_used: usize) -> Result<LicenseInfo> {
    let issued_at = jwt_timestamp_to_utc(raw.iat)?;
    let expires_at = jwt_timestamp_to_utc(raw.exp)?;

    let status = if expires_at < Utc::now() {
        LicenseStatus::Expired
    } else if seats_used > raw.num {
        LicenseStatus::SeatsExceeded
    } else {
        LicenseStatus::Ok
    };

    let license = LicenseInfo {
        r#type: raw.typ,
        status,
        seats: raw.num as i32,
        seats_used: seats_used as i32,
        issued_at,
        expires_at,
    };
    Ok(license)
}

#[async_trait]
impl LicenseService for LicenseServiceImpl {
    async fn read_license(&self) -> Result<Option<LicenseInfo>> {
        let Some(license) = self.db.read_enterprise_license().await? else {
            return Ok(None);
        };
        let license =
            validate_license(&license).map_err(|e| anyhow!("License is corrupt: {e:?}"))?;
        let seats = self.read_used_seats().await?;
        let license = license_info_from_raw(license, seats)?;

        Ok(Some(license))
    }

    async fn update_license(&self, license: Option<String>) -> Result<Option<LicenseStatus>> {
        let mut status = None;
        if let Some(license) = &license {
            let raw =
                validate_license(license).map_err(|e| anyhow!("License is corrupt: {e:?}"))?;
            let seats = self.read_used_seats().await?;
            status = Some(license_info_from_raw(raw, seats)?.status);
        }
        self.db.update_enterprise_license(license).await?;
        Ok(status)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    const VALID_TOKEN: &str = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDUxOTgxMDIsImV4cCI6MTgwNzM5ODcwMiwidHlwIjoiVEVBTSIsIm51bSI6MTB9.vVo7PDevytGw2KXU5E-KMdJBijwOWsD1zKIf26rcjfxa3wDesGY40zuYZWyZFMfmAtBTO7DBgqdWnriHnF_HOnoAEDCycrgoxuSJW5TS9XsCWto-3rDhUsjRZ1wls-ztQu3Gxo_84UHUFwrXe-RHmJi_3w_YO-2L-nVw7JDd5zR8CEdLxeccD47vBrumYA7ybultoDHpHxSppjHlW1VPXavoaBIO1Twnbf52uJlbzJmloViDxoq-_9lxcN1hDN3KKE3crzO9uHK4jjZy_1KNHhCIIcnINek6SBl6lWZw9R88UfdP6uaVOTOHDFbGwv544TSLA_oKZXXntXhldKCp94YN8J4djHim91WwYBQARrpQKiQGP1APEQQdv_YO4iUC3QTLOVw_NMjyma0feVjzHYAap_2Q9HgnxyJfMH-KiH2zaR6BcdOfWV86crO5M0qNoP-XOgy4uU8eE2-PevOKM6uVwYiwoNZL4e9ttH6ratJj0tyqGW_3HYpsVyThzqDPisEz95knsrVL-iagwHRd00l6Mqfwcjbn-gOuUOV9knRIpPvUmfKjjjHgb-JI0qMAIdgeVtwQp0pNqPsKwenMwkpYQH1awfuB_Ia7SyMUNEzTAY8k_J4R6kCZ5XKJ2VTCljd9aJFSZpw-K57reUX1eLc6-Cwt1iI4d23M5UlYjvs";
    const EXPIRED_TOKEN: &str = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDUxOTgxMDIsImV4cCI6MTcwNzM5ODcwMiwidHlwIjoiVEVBTSIsIm51bSI6MTB9.19wrmSSZUQAj_nfnBljUARD3vz_XEIDh4wpi_U2P6LDRcvm7QYCro__LxUjIf45aE9BBiZCPBRTVOw_tMbegTAv5yK9G9cllGPdRDKWjf24BJpHt2wBKOwhCToUKp8R8D50bQ3cxHuz7J3XxcOMtwKxNRlwaufO-vgxX73v13z_bN6y5ix8FC5JEjY1z3fNPc_TnuuHnaXXqgqL9OJTrxhh5FErqR52kmxGGn2KCM8rm2Nfu0It2IZQuyJHSceZ3-iiIxsrVdXxbO4KHXLEOXos0xJRV8QG9_9VjAo6qui6BioygwrcPqHT7OoG3WfcT8XE9rcEX-s9PZ54_XxLm0yh81g54xPI92n94pe32XfE9T-YXNK3MLAdZWwDhp_sKXTcMSIr7mI9OA7eczZUpvI4BuDM8s1irNx4DKdfTwNchHDfEPmGmO53RHyVEbrS72jF9GBRBIwPmpGppWhcwpVNmlRJw3j1Sa_ttcGikPnBZBrUxGqzynq4q1VpeCpRoTzO9_nw5eciKMpaKww0P5Edqm5kKgg48aABfsTU3hLqTIr9rgjXePL_gEse6MJX_JC8I7-R17iQmMxKiNa9bTqSIk56qlB6gwZTzcjEtpnYlzZ05Ci6D3JBH9ZdO_F3UZDt5JdAD5dqsKl8PfWpxaWpg7FXNlqxYO9BpxCwr_7g";
    const INCOMPLETE_TOKEN: &str = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDUxOTgxMDIsImV4cCI6MTgwNzM5ODcwMiwidHlwIjoiVEVBTSJ9.Xdp7Tgi39RN3qBfDAT_RncCDF2lSSouT4fjR0YT8F4qN8qkocxgvCa6JyxlksaiqGKWb_aYJvkhCviMHnT_pnoNpR8YaLvB4vezEAdDWLf3jBqzhlsrCCbMGh72wFYKRIODhIHeTzldU4F06I9sz5HdtQpn42Q8WC8tAzG109vHtxcdC7D85u0CumJ35DcV7lTfpfIkil3PORReg0ysjZNjQ2JbiFqMF1VbBmC-DsoTrJoHlrxdHowMQsXv89C80pchx4UFSm7Z9tHiMUTOzfErScsGJI1VC5p8SYA3N4nsrPn-iup1CxOBIdK57BHedKGpd_hi1AVWYB4zXcc8HzzpqgwHulfaw_5vNvRMdkDGj3X2afU3O3rZ4jT_KLGjY-3Krgol8JHgJYiPXkBypiajFU6rVeMLScx-X-2-n3KBdR4GQ9la90QHSyIQUpiGRRfPhviBFDtAfcjJYo1Irlu6MGVhgFq9JH5SOVTn57V0A_VeAbj8WZNdML9hio9xqxP86DprnP_ApHpO_xbi-sx2GCmUyfC10eKnX8_sAB1n7z0AaHz4e-6SGm1I-wQsWcXjZfRYw0Vtogz7wVuyAIpm8lF58XjtOwQ9bP1kD03TGIcBTvEtgA6QUhRcximGJ5buK9X2TTd4TlHjFF1krrmYAUEDgFsorseoKvMkspVE";

    #[test]
    fn test_validate_license() {
        let license = validate_license(VALID_TOKEN).unwrap();
        assert_eq!(license.iss, "tabbyml.com");
        assert_eq!(license.sub, "fake@tabbyml.com");
        assert_matches!(license.typ, LicenseType::Team);
        assert_eq!(
            license_info_from_raw(license, 11).unwrap().status,
            LicenseStatus::SeatsExceeded
        );
    }

    #[test]
    fn test_expired_license() {
        let license = validate_license(EXPIRED_TOKEN).unwrap();
        let license = license_info_from_raw(license, 0).unwrap();
        assert_matches!(license.status, LicenseStatus::Expired);
    }

    #[test]
    fn test_missing_field() {
        let license = validate_license(INCOMPLETE_TOKEN);
        assert_matches!(
            license,
            Err(jwt::errors::ErrorKind::MissingRequiredClaim(_))
        );
    }

    #[tokio::test]
    async fn test_create_delete_license() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_license_service(db);

        assert!(service
            .update_license(Some("bad_token".into()))
            .await
            .is_err());

        service
            .update_license(Some(VALID_TOKEN.into()))
            .await
            .unwrap();
        assert!(service.read_license().await.unwrap().is_some());

        service.update_license(None).await.unwrap();
        assert!(service.read_license().await.unwrap().is_none());

        service
            .update_license(Some(EXPIRED_TOKEN.into()))
            .await
            .unwrap();
        let info = service.read_license().await.unwrap().unwrap();
        assert_eq!(info.status, LicenseStatus::Expired);
    }
}
