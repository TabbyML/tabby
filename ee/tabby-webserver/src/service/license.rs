use jsonwebtoken as jwt;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref LICENSE_DECODING_KEY: jwt::DecodingKey =
        jwt::DecodingKey::from_rsa_pem(include_bytes!("../../keys/license.key.pub")).unwrap();
}

#[derive(Debug, Deserialize)]
pub enum LicenseType {
    TEAM,
    ENTERPRISE,
}

#[derive(Debug, Deserialize)]
pub enum LicenseAddon {
}

#[derive(Debug, Deserialize)]
pub struct LicenseInfo {
    /// Expiration time (as UTC timestamp)
    pub exp: f64,

    /// Issued at (as UTC timestamp)
    pub iat: f64,

    /// Issuer
    pub iss: String,

    /// License grantee email address
    pub sub: String,

    /// License Type
    pub typ: LicenseType,

    /// License addons.
    #[serde(skip)]
    pub add: Vec<LicenseAddon>,

    /// Count of license (# of seats).
    pub cnt: usize,
}

pub fn validate_license(token: &str) -> jwt::errors::Result<LicenseInfo> {
    let mut validation = jwt::Validation::new(jwt::Algorithm::RS512);
    validation.set_issuer(&["tabbyml.com"]);
    validation.set_required_spec_claims(&["exp", "iat", "sub", "iss", "typ", "add", "cnt"]);
    let data = jwt::decode::<LicenseInfo>(token, &LICENSE_DECODING_KEY, &validation)?;
    Ok(data.claims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;

    #[test]
    fn test_validate_license() {
        let token = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDcyOTgxMDIsImV4cCI6MTkwNzM5ODcwMiwidHlwIjoiVEVBTSIsImNudCI6MTB9.YSx02tyQMJ3v3e4jnOnJ_3V_XEsbyVLzbJJfcgDrlsm-0fYPppH1VI7w3LiNR1Gd3oJZDW5bvKoKyeRpBlxVcFw1ke0CN_IQLFGEmAw-xkVY07_VzaHF8azbfHS4-PHDBzW-DdpGvidxLXT7bYm3JskJDC22z-fTp9xnP799xAIEzyDK4x0Lphc_f3TbTw9kgh2obKsek6HEJ8eJN03CsyIONPS2cakGgOgnJof9Ni7tnRFs0WOdqoQiAKK_F8vd0xsMVA3cGQNsT-Mnd-_XjUyuiommmIB9NXrdg67854x5pvWDP3a4FqfTgwTxqQrtCsMvkKDwQ59fyaEakW_ohvKkJnaCkkRbWrGaoFODI5vghNf7BA_tAR7pxoWt6El-u85YEdW9z5jdlhkGLkr-Xd3GjEg_D8b1T6KKd-8Rz6iIcJp4YhGNJkiA5svkvhvI09QVRcO98e94FAsMkM9t3VJ21mQ8nBbhgMmCzIuufbY3RyNSQfFMt6tKnIuXhwSwvFsVY6ib72B_zd4oEUstTW7F-eBONG_FXOo7-VQ8yttvjtwjvoAmieKz1XP4Luj8DaAFnqXQRqTy2JzoDtP7zyLqtz6wLm-N2GAtpcskbPkmAIJb3cCdhtNNxMqqy4vTRL_xkVv_L2aWjbOdIxr9q9bUfdZ2iYxJWZl8H31uw6I";
        let license = validate_license(token).unwrap();
        assert_eq!(license.iss, "tabbyml.com");
        assert_eq!(license.sub, "fake@tabbyml.com");
        assert_matches!(license.typ, LicenseType::TEAM);
        assert!(license.add.is_empty());
    }

    #[test]
    fn test_expired_license() {
        let token = "eyJhbGciOiJSUzUxMiJ9.eyJpc3MiOiJ0YWJieW1sLmNvbSIsInN1YiI6ImZha2VAdGFiYnltbC5jb20iLCJpYXQiOjE3MDcxOTgxMDIsImV4cCI6MTcwNzM5ODcwMiwidHlwIjoiVEVBTSIsImNudCI6MTB9.rX19CUESvvICiNRv4m9Yhq-xAJ_0XUq9thqV4MRNdmIdvDw4DskCJ0m7dKf1o_t4YTmffCA_-9gVR7O1Yn-Zpe6l717gSOcxQfI3hjReJYhwHvsST1FEWP0G6KbhicuMF6hv-nvLdnXEIVJ6xF_Hga0stDtcZ3Eed1H1y_hBdO7qjKkVQ22nhn6uFEgL9_xm9uWNDxRxcehZrxS4i20O3Q6qlsqWog_rmYjPfMbh8beHwwtkE4FBRB5I-Y_AIXb9NZHCSAiYkSCQTssi8opl6c5KM2BnnAkbiwR10Jg5jVoQBld6VeSmuYIrs82RxdIyEMu3Dx2mjxwyCzIMGAjUAV5EdPvCpoIzCAZ2Qw9DJFSbnq8Tfldle6DcpdNqWPH4I7V4ecZvW3E-vLUN3saAhbvvhwqIxXiZsxKneVKfp7Cs-9mjQz5Hm02KtfLtnnL1_LzgA8p2lHGXTTpvLZxztJEgdapZRMs34QA4ERvcBmYQUgARymVxQdey9uRJJB_7r27YW-gr-_KOtW8zfJULV7tnigUjhXooyTovyj7KU8FSwBf7yiP0aD4dzkdF8Sva33ELABSPEUpl3HT8NjdtryQQf763Ua9Gs1BaNFcsup0DjkQO1yEFp6qPxLj4DfIEOOSNRRhQk-mb2N60Qh8GxKMoi8tkL3CzPVOrqlFMwEE";
        let license = validate_license(token);
        assert!(license.is_err());
        let err = license.unwrap_err();
        assert_eq!(err.kind(), &jwt::errors::ErrorKind::ExpiredSignature);
    }
}
