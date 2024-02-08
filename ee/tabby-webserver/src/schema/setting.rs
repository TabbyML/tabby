use std::collections::HashSet;

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
    #[validate(url)]
    pub external_url: String,
}

fn validate_unique_domains(domains: &[String]) -> Result<(), ValidationError> {
    let unique: HashSet<_> = domains.iter().collect();
    if unique.len() != domains.len() {
        let collision = domains.iter().find(|s| unique.contains(s)).unwrap();
        let mut err = ValidationError::new("securityAllowedRegisterDomainList");
        err.message = Some(format!("Duplicate domain: {collision}").into());
        return Err(err);
    }
    for domain in domains {
        let email = format!("noreply@{domain}");
        if !validate_email(email) {
            let mut err = ValidationError::new("securityAllowedRegisterDomainList");
            err.message = Some(format!("Invalid domain name: {domain}").into());
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
        assert!(validate_unique_domains(&vec!["example.com".to_owned()]).is_ok());

        assert!(validate_unique_domains(&vec!["https://example.com".to_owned()]).is_err());

        assert!(validate_unique_domains(&vec!["domain.withmultipleparts.com".to_owned()]).is_ok());

        assert!(validate_unique_domains(&vec!["example.com".to_owned(), "example.com".to_owned()]).is_err());
    }
}
