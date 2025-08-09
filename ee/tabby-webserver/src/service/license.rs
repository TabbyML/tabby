use anyhow::{anyhow, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use jsonwebtoken as jwt;
use lazy_static::lazy_static;
use serde::Deserialize;
use tabby_db::DbConn;
use tabby_schema::{
    is_demo_mode,
    license::{LicenseFeature, LicenseInfo, LicenseService, LicenseStatus, LicenseType},
    Result,
};

use crate::bail;

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

    /// Features
    #[serde(default)]
    pub fet: Vec<LicenseFeature>,

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
    Ok(DateTime::from_timestamp(secs, 0).context("Timestamp is corrupt")?)
}

struct LicenseServiceImpl {
    db: DbConn,
}

impl LicenseServiceImpl {
    async fn make_community_license(&self) -> Result<LicenseInfo> {
        let seats_used = self.db.count_active_users().await?;
        let status = if seats_used > LicenseInfo::seat_limits_for_community_license() {
            LicenseStatus::SeatsExceeded
        } else {
            LicenseStatus::Ok
        };

        Ok(LicenseInfo {
            r#type: LicenseType::Community,
            status,
            seats: LicenseInfo::seat_limits_for_community_license() as i32,
            seats_used: seats_used as i32,
            issued_at: None,
            expires_at: None,
            features: None,
        }
        .guard_seat_limit())
    }

    async fn make_demo_license(&self) -> Result<LicenseInfo> {
        let seats_used = self.db.count_active_users().await? as i32;
        Ok(LicenseInfo {
            r#type: LicenseType::Enterprise,
            status: LicenseStatus::Ok,
            seats: 100,
            seats_used,
            issued_at: None,
            expires_at: None,
            features: Some(vec![LicenseFeature::CustomLogo]),
        })
    }
}

pub async fn new_license_service(db: DbConn) -> Result<impl LicenseService> {
    Ok(LicenseServiceImpl { db })
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
        issued_at: Some(issued_at),
        expires_at: Some(expires_at),
        features: Some(raw.fet),
    }
    .guard_seat_limit();
    Ok(license)
}

#[async_trait]
impl LicenseService for LicenseServiceImpl {
    async fn read(&self) -> Result<LicenseInfo> {
        if is_demo_mode() {
            return self.make_demo_license().await;
        }

        let Some(license) = self.db.read_enterprise_license().await? else {
            return self.make_community_license().await;
        };
        let license =
            validate_license(&license).map_err(|e| anyhow!("License is corrupt: {e:?}"))?;
        let seats = self.db.count_active_users().await?;
        let license = license_info_from_raw(license, seats)?;

        Ok(license)
    }

    async fn update(&self, license: String) -> Result<()> {
        if is_demo_mode() {
            bail!("Modifying license is disabled in demo mode");
        }

        let raw = validate_license(&license).map_err(|_e| anyhow!("License is not valid"))?;
        let seats = self.db.count_active_users().await?;
        match license_info_from_raw(raw, seats)?.status {
            LicenseStatus::Ok => self.db.update_enterprise_license(Some(license)).await?,
            LicenseStatus::Expired => bail!("License is expired"),
            LicenseStatus::SeatsExceeded => {
                bail!("License doesn't contain sufficient number of seats")
            }
        };
        Ok(())
    }

    async fn reset(&self) -> Result<()> {
        self.db.update_enterprise_license(None).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    const VALID_TOKEN: &str = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDUxOTgxMDIsImV4cCI6MTgwNzM5ODcwMiwidHlwIjoiVEVBTSIsIm51bSI6MX0.r99qAkHGAzjZtS904ko5MMklquMcEJdibVGAZAxrJTf-kKBT-Kc-u-A8o7ZSrLD0eubIxNrLb16UsyAMxJ6xnIJY4h8BTIR9cz_dTezyGywpuAKI13Q2S77tfwcyBF6icFkDsz187MSQGPQuTdVNU8zXkYR5ZkNs8_Uc8SL940xt0KHWLU9DX8KT6eCcVMwAypLyAsSTRJeqE8uRumq1K6dKK7wkE_HQrg9nSmr40A5ZZPzRsUp6hShJyMYSp-D02utbT8bAzVPw6alBgZWrmlVEvdcvfO81DZylUIm-pszKityfT5tmuyMWtUx3AeLXSiQWZOpah3OBnL11IKhNhYWSzUMGuDENHfbP9hlSJvzjq8WeN73nXSjkNEVYetT2er6pnoGrvFUBWcLLdWcl4p324WwqsP5A7ZDbWamo62yPxHUy7Vr4ySRLDfNEQbjP8JVPacpx3-5oY16LlzS4e9RhR0G-aykJitrLd5--gTVGxlxsLbmz33TTDd3nMGuQp2xmpZsw4rTKefEN7hCdvgJhtwRLgL4jxSm2mBgtwWH_i0uuBFpCYNgh97rU-Cak66adXDydAOr6-imSHAIlSphGj6G4rUdbMtBV0n1MVGg3vIyHQot3hMaH6uXMpHOUEtxQivkp0F-fY6PoFr49HfWD-ZuneENaKKjB8p_rd9k";
    const EXPIRED_TOKEN: &str = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDUxOTgxMDIsImV4cCI6MTcwNDM5ODcwMiwidHlwIjoiVEVBTSIsIm51bSI6MX0.UBufd2YlyhuChdCSZvbvEBtxLABhZSuhya4KHKHYM2ABaSTjYYtSyT-yv0i9b8sySBoeu7kG0XBNrLQOg4fcirR5DxOFxiskI7qLLSQEIDYe-xnEbvxqKhN3RpHkxik9_OlvElvpIGrZRQxiELhESIM0NGck0Dz6MwTDFutkHZFh06cLFeoihs1rn44SknL3wP_afyCaOpQtTjDfsayBMfyDAriTG8HSnPbrw5Om7ER7uAqszhX8wpFonDeFeVB0OIUjayfL-SAMdLqNEqaFsUcuE4cUk7o9tA2jsYz2-BRlwDocLpRVp2V-K8MuyQJhDTiswbey2DE5tNRvnd3nNaVr7Pmt3mF7NMt8op8hl4I9scoThFBj9Bb1iMfAGVSXlRn9Kf2HHe2BJXGWC3w9bjWH2KRPMP3tScJ4CQccIJxZPU-fcX7IC1q8R4PWDYS11TDJ03PvCTEGFt3fBTLLaGOeoYHYNnd4qux317YhGtWTOO6ESIuoxQkJdTpNVOwfNmCVSfFUvJYs0l4r7z-QouHAd79Ck_GJ-cdiIOrV9MB1Lq6ayk267bXfdi0Lx6-PYxrTwXEkF5tBydrsPyhoReAbH8yQDqzlPbQzOlLo--Z4940kSEpgEsL9G6ymG5wDlMzNuQfjbYbCI0L19Spx5QRGtyYXtiSU1Tq-hhGm3zA";
    const INCOMPLETE_TOKEN: &str = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDUxOTgxMDIsImV4cCI6MTgwNDM5ODcwMiwidHlwIjoiVEVBTSJ9.juNQeg8jMRj7Q2XbmHSdneKZbTP_BIL43yW3He5avIRAKee1NF9-qg4ndGOYVWBmtoO6Y_CAts_trSw6gmuDuwWcmSbbr7CWQOYuNrMj1_Gp1MctA8zzC3yzr0EoBLzqkNBq3OySlfOkohopmJ6Lu0d0KRtf46qq94cMDAlfs7etcVGkGqfMEwxznptXiF7_S3qRVbahvJDPJlu_ozwn51tICXMrlGV_P6jdBcNLQ8I1LAH2RfyH9u-4mUSTKt-obnXw6mtPxPjl07MEajM_wW3X05-iRygQfyzDulvW0EXf39OnW2kCuyfQWx5Zksr-sCNTEL2VSalf9o8MchjAhDN5QrygdZkk7KXwt3O54tpcnFVABw9ORxJtTrsZJD-YvdmS01O6qLfMRWs2CGWFTfDJLxMSiBhAsy4DC4TkZN4UnBpX09U7n6f_0NUr83YAWcw0Rlp32k01j9iPUWSdePZh46Ck00XdzLcc15xfqv__ilaLAyRtb9JUVBX7g-VaLb1YGk658t19eukRNzE6WFyKfAE7u6EbxowtFQqVKYXWX_zDHoalo3DjUmPBV_VsorcBg4cjhrhBPBOB5f7Wa8r7eiJz1gWEj1xJEK2Y_mdShAvxNSWPSTvNvviPTgJbvbwDTzQ0It_d066ADBY2o0y5DTMP23EPL-oZ14TYIY4";

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
    async fn test_license_mutations() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_license_service(db).await.unwrap();

        assert!(service.update("bad_token".into()).await.is_err());

        service.update(VALID_TOKEN.into()).await.unwrap();
        assert!(service.read().await.is_ok());

        assert!(service.update(EXPIRED_TOKEN.into()).await.is_err());

        service.reset().await.unwrap();
        assert_eq!(
            service.read().await.unwrap().seats,
            LicenseInfo::seat_limits_for_community_license() as i32
        );
    }
}
