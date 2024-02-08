use std::collections::{HashMap, HashSet};

use anyhow::Result;
use async_trait::async_trait;
use juniper::{GraphQLInputObject, GraphQLObject};
use validator::{validate_email, Validate, ValidationError};

#[async_trait]
pub trait SettingService: Send + Sync {
    async fn read_security_setting(&self) -> Result<SecuritySetting>;
    async fn update_security_setting(&self, input: SecuritySettingInput) -> Result<()>;

    async fn read_network_setting(&self) -> Result<NetworkSetting>;
    async fn update_network_setting(&self, input: NetworkSettingInput) -> Result<()>;
}

#[derive(GraphQLObject, Debug, PartialEq)]
pub struct SecuritySetting {
    pub allowed_register_domain_list: Vec<String>,
    pub disable_client_side_telemetry: bool,
}

#[derive(GraphQLInputObject, Validate)]
pub struct SecuritySettingInput {
    #[validate(custom = "validate_unique_domains")]
    pub allowed_register_domain_list: Vec<String>,
    pub disable_client_side_telemetry: bool,
}

#[derive(GraphQLObject, Debug, PartialEq)]
pub struct NetworkSetting {
    pub external_url: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct NetworkSettingInput {
    #[validate(url(code = "externalUrl", message = "URL is malformed"))]
    pub external_url: String,
}

fn validate_unique_domains(domains: &[String]) -> Result<(), ValidationError> {
    let unique: HashSet<_> = domains.iter().collect();
    if unique.len() != domains.len() {
        let i = domains.iter().position(|s| unique.contains(s)).unwrap();
        let err = ValidationError {
            code: format!("allowedRegisterDomainList.{i}.value").into(),
            message: Some("Duplicate domain".into()),
            params: HashMap::default(),
        };
        return Err(err);
    }
    for (i, domain) in domains.iter().enumerate() {
        let email = format!("noreply@{domain}");
        if !validate_email(email) {
            let err = ValidationError {
                code: format!("allowedRegisterDomainList.{i}.value").into(),
                message: Some("Invalid domain".into()),
                params: HashMap::default(),
            };
            return Err(err);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::schema::setting::validate_unique_domains;

    #[test]
    fn test_validate_urls() {
        assert!(validate_unique_domains(&["example.com".to_owned()]).is_ok());

        assert!(validate_unique_domains(&["https://example.com".to_owned()]).is_err());

        assert!(validate_unique_domains(&["domain.withmultipleparts.com".to_owned()]).is_ok());

        assert!(
            validate_unique_domains(&["example.com".to_owned(), "example.com".to_owned()]).is_err()
        );
    }
}
